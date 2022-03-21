import torch
import torch.nn as nn
import torch.nn.functional as F


class MAMIL1D(nn.Module):
    def __init__(self, n_templates: int = 10, bottleneck_width: int = 4):
        """Initializes 2D MAMIL model.

        Args:
          n_templates: Number of templates.
          bottleneck_width: Bottleneck spatial width (and height), depends on input patches size.
        """
        super().__init__()
        self.L = 128
        self.L2 = self.L * 2
        self.D = 128
        self.K = 1
        self.n_templates = n_templates
        self.bottleneck_width = bottleneck_width
        self.channels = 50
        self.embedding_dim = self.channels * self.bottleneck_width ** 2

        self.feature_extractor_part1 = nn.Sequential(
            nn.Conv2d(1, 20, kernel_size=5),
            nn.ReLU(),
            nn.MaxPool2d(2, stride=2),
            nn.Conv2d(20, self.channels, kernel_size=5),
            nn.ReLU(),
            nn.MaxPool2d(2, stride=2)
        )

        self.feature_extractor_part2 = nn.Sequential(
            nn.Linear(self.embedding_dim, self.L),
            nn.ReLU(),
        )

        self.neighbours_attention = nn.Sequential(
            nn.Linear(self.L, self.L),
            nn.Tanh()
        )

        self.templates = nn.Parameter(torch.randn(self.n_templates, self.L2,
                                                   requires_grad=True))
        self.proto_attention = nn.Sequential(
            nn.Linear(self.L2, self.L2),
            nn.Tanh()
        )
        self.global_attention = nn.Sequential(
            nn.Linear(self.L2, self.D),
            nn.Tanh(),
            nn.Linear(self.D, self.K)
        )

        self.classifier = nn.Sequential(
            nn.Linear(self.L2*self.K, 1),
            nn.Sigmoid()
        )

    def forward(self, x, return_proto_scores=False):
        """Calculates instance-level probabilities and an attention matrix.

        Args:
          x: Input batch with exactly one bag of batches.
          return_proto_scores: Return templates scores or not.

        Returns:
          Tuple of (instance probabilities, instance labels, attention matrix),
          or (instance probabilities, instance labels, attention matrix, patch scores, template scores)
          if `return_proto_scores` is True.
            
        """
        x = x.squeeze(0)

        H = self.feature_extractor_part1(x)
        H = H.view(-1, self.embedding_dim)
        H = self.feature_extractor_part2(H)  # NxL

        # Classical Attention-MIL:
        # A = self.attention(H)  # NxK
        # A = torch.transpose(A, 1, 0)  # KxN
        # A = F.softmax(A, dim=1)  # softmax over N
        # M = torch.mm(A, H)  # KxL

        # Neighbourhood attention:
        #   H shape: NxL
        #   Naive loop-based implementation:
        neighbourhood_embeddings = []
        for i in range(H.shape[0]):
            if i == 0:
                cur_neighbours = torch.cat((H[i + 1: i + 2], H[i + 1: i + 2]),
                                           axis=0)
            elif i == H.shape[0] - 1:
                cur_neighbours = torch.cat((H[i - 1: i], H[i - 1: i]),
                                           axis=0)
            else:
                cur_neighbours = torch.cat((H[i - 1: i], H[i + 1: i + 2]),
                                           axis=0)
            cur_instance_embedding = H[i: i + 1]
            # cur_neighbours shape: 2xL
            cur_alphas = torch.mm(
                self.neighbours_attention(cur_neighbours),
                cur_instance_embedding.T
            )  # 2x1
            cur_alphas = torch.transpose(cur_alphas, 1, 0)  # 1x2
            cur_alphas = F.softmax(cur_alphas, dim=1)  # 1x2
            cur_neighbourhood_emb = torch.mm(cur_alphas, cur_neighbours)  # 1xL
            neighbourhood_embeddings.append(cur_neighbourhood_emb)
        neighbourhood_embeddings = torch.cat(neighbourhood_embeddings, dim=0)
        H = torch.cat((H, neighbourhood_embeddings), dim=1)  # Nx2L

        # Multi-template:
        patch_scores = torch.mm(
            self.proto_attention(H), # H,
            self.templates.T
        )  # NxP, P = n_templates
        patch_scores = torch.transpose(patch_scores, 1, 0)  # PxN
        betas = F.softmax(patch_scores, dim=1)  # PxN
        embs = torch.mm(betas, H)  # PxL2

        template_scores = self.global_attention(embs)  # PxK
        template_scores = torch.transpose(template_scores, 1, 0)  # KxP
        gammas = F.softmax(template_scores, dim=1)  # KxP
        M = torch.mm(gammas, embs)  # KxL2
        A = torch.mm(gammas, betas)

        Y_prob = self.classifier(M)
        Y_hat = torch.ge(Y_prob, 0.5).float()

        if return_proto_scores:
            return Y_prob, Y_hat, A, patch_scores, template_scores
        return Y_prob, Y_hat, A

    def calculate_classification_error(self, X, Y):
        """Calculates classification error.
        
        Args:
          X: Input batch consisting of one bag (see `forward` for the details).
          Y: Groung truth bag label.
        
        Returns:
          Tuple of (binary error, predicted instance labels)
        """
        Y = Y.float()
        _, Y_hat, _ = self.forward(X)
        error = 1. - Y_hat.eq(Y).cpu().float().mean().item()

        return error, Y_hat

    def calculate_objective(self, X, Y):
        """Calculates training objective.
        
        Args:
          X: Input batch consisting of one bag (see `forward` for the details).
          Y: Groung truth bag label.

        Returns:
          Tuple of (loss tensor, attention matrix).
        """
        Y = Y.double()
        Y_prob, _, A = self.forward(X)
        Y_prob = torch.clamp(Y_prob, min=1e-5, max=1. - 1e-5)
        neg_log_likelihood = -1. * (Y * torch.log(Y_prob) + (1. - Y) * torch.log(1. - Y_prob))  # negative log bernoulli
        return neg_log_likelihood, A


def _is_neighbour(a, b, distance: int = 1):
    return 0 < max(abs(a[0] - b[0]), abs(a[1] - b[1])) <= distance


class MAMIL2D(nn.Module):
    def __init__(self, n_templates: int = 10,
                 use_neighbourhood: bool = True,
                 bottleneck_width: int = 6):
        """Initializes 2D MAMIL model.

        Args:
          n_templates: Number of templates.
          use_neighbourhood: Use neighbourhood attention.
          bottleneck_width: Bottleneck spatial width (and height), depends on input patches size.
        """
        super().__init__()
        self.n_templates = n_templates
        self.use_neighbourhood = use_neighbourhood
        self.L = 512
        if self.use_neighbourhood:
            self.L2 = self.L * 2
        else:
            self.L2 = self.L
        self.D = 128
        self.K = 1
        self.bottleneck_width = bottleneck_width
        self.channels = 48
        self.embedding_dim = self.channels * self.bottleneck_width ** 2

        self.feature_extractor_part1 = nn.Sequential(
            nn.Conv2d(3, 36, kernel_size=4),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, stride=2),
            nn.Conv2d(36, self.channels, kernel_size=3),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, stride=2),
        )

        self.feature_extractor_part2 = nn.Sequential(
            nn.Linear(self.embedding_dim, self.L),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(self.L, self.L),
            nn.ReLU(inplace=True),
            nn.Dropout(),
        )

        # self.attention = nn.Sequential(
        #     nn.Linear(self.L, self.D),
        #     nn.Tanh(),
        #     nn.Linear(self.D, self.K)
        # )

        self.neighbours_attention = nn.Sequential(
            nn.Linear(self.L, self.L),
            nn.Tanh()
        )

        self.templates = nn.Parameter(torch.randn(self.n_templates, self.L2,
                                                   requires_grad=True))
        self.proto_attention = nn.Sequential(
            nn.Linear(self.L2, self.L2),
            nn.Tanh()
        )
        self.global_attention = nn.Sequential(
            nn.Linear(self.L2, self.D),
            nn.Tanh(),
            nn.Linear(self.D, self.K)
        )

        self.classifier = nn.Sequential(
            nn.Linear(self.L2*self.K, 1),
            nn.Sigmoid()
        )

    def forward(self, x, positions=None, temperature=1.0):
        """Calculates instance-level probabilities and an attention matrix.

        Args:
          x: Input batch with exactly one bag of batches.
          positions: Patches positions.
          temperature: Softmax temperature.

        Returns:
          Tuple of (instance probabilities, instance labels, attention matrix)
        """
        x = x.squeeze(0)

        H = self.feature_extractor_part1(x)
        H = H.view(-1, self.embedding_dim)
        H = self.feature_extractor_part2(H)  # NxL

        # Classical Attention-MIL:
        # A = self.attention(H)  # NxK
        # A = torch.transpose(A, 1, 0)  # KxN
        # A = F.softmax(A, dim=1)  # softmax over N
        # M = torch.mm(A, H)  # KxL

        # Neighbourhood attention:
        #   H shape: NxL
        #   Naive loop-based implementation:
        if self.use_neighbourhood:
            assert positions is not None
            
            neighbourhood_embeddings = []
            # for i in range(H.shape[0]):
            assert len(positions) == H.shape[0]
            for i, pos in enumerate(positions):
                nbrs = [j for j, jpos in enumerate(positions) if j != i and _is_neighbour(jpos, pos)]
                if len(nbrs) > 0:
                    cur_neighbours = torch.cat([H[j:j + 1] for j in nbrs], axis=0)
                else:
                    cur_neighbours = H[i: i + 1]
                cur_instance_embedding = H[i: i + 1]
                # cur_neighbours shape: 2xL
                cur_alphas = torch.mm(
                    self.neighbours_attention(cur_neighbours),
                    cur_instance_embedding.T
                )  # 2x1
                cur_alphas = torch.transpose(cur_alphas, 1, 0)  # 1x2
                cur_alphas = F.softmax(cur_alphas, dim=1)  # 1x2
                cur_neighbourhood_emb = torch.mm(cur_alphas, cur_neighbours)  # 1xL
                neighbourhood_embeddings.append(cur_neighbourhood_emb)
            # raise Exception()
            neighbourhood_embeddings = torch.cat(neighbourhood_embeddings, dim=0)
            H = torch.cat((H, neighbourhood_embeddings), dim=1)  # Nx2L

        # Multi-template:
        betas = torch.mm(
            self.proto_attention(H), # H,
            self.templates.T
        )  # NxP, P = n_templates
        betas = torch.transpose(betas, 1, 0)  # PxN
        betas = F.softmax(betas / temperature, dim=1)  # PxN
        embs = torch.mm(betas, H)  # PxL2

        gammas = self.global_attention(embs)  # PxK
        gammas = torch.transpose(gammas, 1, 0)  # KxP
        gammas = F.softmax(gammas / temperature, dim=1)  # KxP
        M = torch.mm(gammas, embs)  # KxL2

        A = torch.mm(gammas, betas)

        Y_prob = self.classifier(M)
        Y_hat = torch.ge(Y_prob, 0.5).float()

        return Y_prob, Y_hat, A

    def calculate_classification_error(self, X, Y, fwd=None):
        """Calculates classification error.
        
        Args:
          X: Input batch consisting of one bag (see `forward` for the details).
          Y: Groung truth bag label.
          fwd: Result of `self.forward(X)`.
        
        Returns:
          Tuple of (binary error, predicted instance labels)
        """
        Y = Y.float()
        if fwd is None:
            fwd = self.forward(X)
        _, Y_hat, _ = fwd
        error = 1. - Y_hat.eq(Y).cpu().float().mean().item()

        return error, Y_hat

    def calculate_objective(self, X, Y, fwd=None):
        """Calculates training objective.
        
        Args:
          X: Input batch consisting of one bag (see `forward` for the details).
          Y: Groung truth bag label.
          fwd: Result of `self.forward(X)`.

        Returns:
          Tuple of (loss tensor, attention matrix).
        """
        Y = Y.double()
        if fwd is None:
            fwd = self.forward(X)
        Y_prob, _, A = fwd
        loss_fn = nn.BCELoss()
        neg_log_likelihood = loss_fn(Y_prob.view(-1), Y)
        return neg_log_likelihood, A
