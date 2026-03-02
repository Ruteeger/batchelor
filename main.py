import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import pygame
import sys
from keras.datasets import mnist
import numpy as np
import random as rnd

class Trainer:
    def __init__(self, model, optimizer, loss_fn, device =None):
        #self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.device = "cpu"
        self.model = model.to(self.device)
        self.optimizer = optimizer
        self.loss_fn = loss_fn

    def train_batch(self, dataloader):
        self.model.train()
        total_loss = 0.0
        self.optimizer.zero_grad()
        for inputs, targets in dataloader:
            inputs = inputs.to(self.device)
            targets = targets.to(self.device)
            outputs = self.model(inputs)
            loss = self.loss_fn(outputs, targets)
            loss.backward()
            total_loss += loss.item()
        self.optimizer.step()
        return total_loss / len(dataloader)

    @torch.no_grad()
    def validate(self, dataloader):
        self.model.eval()
        total_loss = 0.0
        for inputs, targets in dataloader:
            inputs = inputs.to(self.device)
            targets = targets.to(self.device)
            outputs = self.model(inputs)
            loss = self.loss_fn(outputs, targets)
            total_loss += loss.item()
        return total_loss / len(dataloader)

class AutoEncoder(nn.Module):
    def __init__(self, encoder, decoder):
        super(AutoEncoder, self).__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, x):
        return self.decoder(self.encoder(x))

(train_X, train_y), (test_X, test_y) = mnist.load_data()
model_encoder = nn.Sequential(
    nn.Flatten(0),
    nn.Linear(784, 32)
)
model_decoder = nn.Sequential(
    nn.Linear(32, 784)
)
model = AutoEncoder(model_encoder, model_decoder)
optimizer = torch.optim.Adam(model.parameters(), lr=0.004)
trainer = Trainer(model, optimizer, nn.MSELoss())

training_data = []
for x in train_X:
    training_data.append((torch.tensor(x, dtype=torch.float), torch.flatten(torch.tensor(x, dtype=torch.float))))

    for step in range(1200):
        batch = rnd.sample(training_data, 200)
        loss = trainer.train_batch(batch)
        print(f"batch: {step}, loss = {loss}")









pygame.init()
screen = pygame.display.set_mode((600, 400))
pygame.display.set_caption("Two Grayscale Matrices")

def draw_matrix(scrn, matrix, x, y, size):
    rows, cols = matrix.shape
    for row in range(rows):
        for col in range(cols):
            value = int(matrix[row][col])
            gray = max(0, min(255, value))
            color = (gray, gray, gray)
            rect = pygame.Rect(x + col * size, y + row * size, size, size)
            pygame.draw.rect(scrn, color, rect)

clock = pygame.time.Clock()
running = True

matrix1 = train_X[0]
matrix2 = model(torch.tensor(matrix1, dtype=torch.float)).reshape(28, 28)

while running:
    clock.tick(60)

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        if event.type == pygame.KEYDOWN:
            if event.key == 13: # enter
                matrix1 = rnd.choice(train_X)
                matrix2 = model(torch.tensor(matrix1, dtype=torch.float)).reshape(28, 28)
    screen.fill((255, 255, 255))
    draw_matrix(screen, matrix1, 5, 5, 5)
    draw_matrix(screen, matrix2, 5 + 28 * 5 + 5, 5, 5)
    pygame.display.flip()

pygame.quit()
sys.exit()
