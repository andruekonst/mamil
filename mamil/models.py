import torch
import torch.nn as nn
import torch.nn.functional as F


class PrototypesAttention(nn.Module):
    def __init__(self):
        super().__init__()
        # self.L = 500
        self.L = 128
        self.L2 = self.L * 2
        self.D = 128
        self.K = 1
        self.n_prototypes = 10

        self.feature_extractor_part1 = nn.Sequential(
            nn.Conv2d(1, 20, kernel_size=5),
            nn.ReLU(),
            nn.MaxPool2d(2, stride=2),
            nn.Conv2d(20, 50, kernel_size=5),
            nn.ReLU(),
            nn.MaxPool2d(2, stride=2)
        )

        self.feature_extractor_part2 = nn.Sequential(
            nn.Linear(50 * 4 * 4, self.L),
            nn.ReLU(),
        )

        self.neighbours_attention = nn.Sequential(
            nn.Linear(self.L, self.L),
            nn.Tanh()
        )

        self.prototypes = nn.Parameter(torch.randn(self.n_prototypes, self.L2,
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
        x = x.squeeze(0)

        H = self.feature_extractor_part1(x)
        H = H.view(-1, 50 * 4 * 4)
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

        # Multi-prototype:
        patch_scores = torch.mm(
            self.proto_attention(H), # H,
            self.prototypes.T
        )  # NxP, P = n_prototypes
        patch_scores = torch.transpose(patch_scores, 1, 0)  # PxN
        betas = F.softmax(patch_scores, dim=1)  # PxN
        embs = torch.mm(betas, H)  # PxL2

        prototype_scores = self.global_attention(embs)  # PxK
        prototype_scores = torch.transpose(prototype_scores, 1, 0)  # KxP
        gammas = F.softmax(prototype_scores, dim=1)  # KxP
        M = torch.mm(gammas, embs)  # KxL2
        A = torch.mm(gammas, betas)

        Y_prob = self.classifier(M)
        Y_hat = torch.ge(Y_prob, 0.5).float()

        if return_proto_scores:
            return Y_prob, Y_hat, A, patch_scores, prototype_scores
        return Y_prob, Y_hat, A

    def calculate_classification_error(self, X, Y):
        Y = Y.float()
        _, Y_hat, _ = self.forward(X)
        error = 1. - Y_hat.eq(Y).cpu().float().mean().item()

        return error, Y_hat

    def calculate_objective(self, X, Y):
        Y = Y.double()
        Y_prob, _, A = self.forward(X)
        Y_prob = torch.clamp(Y_prob, min=1e-5, max=1. - 1e-5)
        neg_log_likelihood = -1. * (Y * torch.log(Y_prob) + (1. - Y) * torch.log(1. - Y_prob))  # negative log bernoulli
        return neg_log_likelihood, A