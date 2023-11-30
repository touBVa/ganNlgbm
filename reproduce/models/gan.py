import torch
import torch.nn as nn
import torch.optim as optim

# 생성자 모델 정의
class Generator(nn.Module):
    def __init__(self, z_size):
        super(Generator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(z_size, 512), # 차수를 매우 조밀하게 변환. 데이터군의 관계는 유지하면서 사용이 쉽도록 선형변환
            nn.LayerNorm(512), # 벡터별 특성 뽑아내 정규화-cov 편이 방지
            nn.LeakyReLU(0.2), # negative slope = 0.2
            nn.Linear(512, 128),
            nn.LayerNorm(128),
            nn.LeakyReLU(0.2),
            nn.Linear(128, 32),
            nn.LayerNorm(32),
            nn.LeakyReLU(0.2),
            nn.Linear(32, z_size)
        )

        # 처음 시작하L는 인풋 차원은 차수가 별로 상관없고, 아웃풋이 89이기만 하면 된다.
        # 의문. 이상거래라고 판단하기 위해 GAN을 사용해 오버샘플링이 적절한가? 아무 난수나 넣었는데 우연히 이상거래 셋이 나오는 건 아니고?
        # 즉, GAN이 정말로 필요했는가?

    def forward(self, z):
        return self.model(z)

# 판별자 모델 정의
class Discriminator(nn.Module):
    def __init__(self, z_dim):
        super(Discriminator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(z_dim, 512),
            nn.LayerNorm(512),
            nn.LeakyReLU(0.2),
            nn.Linear(512, 128),
            nn.LayerNorm(512),
            nn.LeakyReLU(0.2),
            nn.Linear(128, 32),
            nn.LayerNorm(32),
            nn.LeakyReLU(0.2),
            nn.Linear(32, 8),
            nn.LayerNorm(8),
            nn.LeakyReLU(0.2)
        )

    def forward(self, x):
        return self.model(x)

# 생성자, 판별자, 손실 함수, 최적화 함수 초기화
generator = Generator()
discriminator = Discriminator()
criterion = nn.BCELoss()
optimizer_g = optim.Adam(generator.parameters(), lr=0.0002)
optimizer_d = optim.Adam(discriminator.parameters(), lr=0.0002)

# 학습 예제
num_epochs = 10000
for epoch in range(num_epochs):
    for _ in range(100):  # 100개의 랜덤한 샘플 생성
        noise = torch.randn(10, 89)  # 10개의 샘플, 89개의 차원
        generated_data = generator(noise)
        real_data = torch.randn(10, 89)  # 실제 데이터 (여기서는 랜덤하게 생성)

        # 판별자 학습
        optimizer_d.zero_grad()
        d_real = discriminator(real_data)
        d_fake = discriminator(generated_data.detach())
        loss_d_real = criterion(d_real, torch.ones_like(d_real))
        loss_d_fake = criterion(d_fake, torch.zeros_like(d_fake))
        loss_d = loss_d_real + loss_d_fake
        loss_d.backward()
        optimizer_d.step()

    # 생성자 학습
    optimizer_g.zero_grad()
    d_fake = discriminator(generated_data)
    loss_g = criterion(d_fake, torch.ones_like(d_fake))
    loss_g.backward()
    optimizer_g.step()

    # 일정 주기로 손실 출력
    if epoch % 500 == 0:
        print(f"Epoch [{epoch}/{num_epochs}], Loss D: {loss_d.item()}, Loss G: {loss_g.item()}")
