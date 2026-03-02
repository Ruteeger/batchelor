"""
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import pygame
import sys
from keras.datasets import mnist

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
"""
import numpy as np
import random as rnd

#train_X = [np.sin(np.arange(0, 10, 0.4) + rnd.random()) for i in range(100)]
train_X = [np.arange(0, 3, 1) * (1 + rnd.random()) + 7 * rnd.random() for i in range(20)]
data_dim = len(train_X[0])

encode_dim = 2
A = np.random.random((encode_dim, data_dim)) / np.sqrt(encode_dim * data_dim)
B = np.random.random((data_dim, encode_dim)) / np.sqrt(encode_dim * data_dim)

avg_x = np.average(train_X)
covariance_matrix = np.zeros((data_dim, data_dim))
for x in train_X:
    x_tilde = x - avg_x
    covariance_matrix += np.outer(x_tilde, x_tilde)
covariance_matrix /= len(train_X)

def cost():
    sum = 0.0
    for x in train_X:
        x_tilde = x - avg_x
        sum += np.linalg.norm(B @ A @ x_tilde - x_tilde)
    return sum

for step in range(5000):
    temp = B @ A @ covariance_matrix - covariance_matrix
    dA = B.T @ temp
    B -= 0.02 * temp @ A.T
    A -= 0.02 * dA
print(B @ A)
print(cost())
"""
pygame.init()
screen = pygame.display.set_mode((600, 400))
pygame.display.set_caption("Two Grayscale Matrices")

def draw_matrix(scrn, matrix, size, x, y):
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

matrix1 = np.array(train_X[0])
matrix2 = (B @ A @ np.array(train_X[0]).flatten()).reshape(28, 28)
print(matrix2)

while running:
    clock.tick(60)

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
    screen.fill((255, 255, 255))
    draw_matrix(screen, matrix1, 5, 5, 5)
    draw_matrix(screen, matrix2, 5, 5 + 29 * 5, 5)
    pygame.display.flip()

pygame.quit()
sys.exit()
"""

"""
    model = nn.Sequential(
        nn.Linear(28 * 28, 32),
        nn.ReLU(),
        nn.Linear(32, 32),
        nn.ReLU(),
        nn.Linear(32, 10),
        nn.Sigmoid()
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.CrossEntropyLoss()
    trainer = Trainer(model, optimizer, loss_fn)
    print(model(torch.flatten(torch.tensor(train_X[0], dtype=torch.float) / 256.0)))
"""
