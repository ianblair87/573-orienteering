class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()

        self.features = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 1, kernel_size=5, padding=2),
            nn.Sigmoid()
        )

        def weights_init(m):
          if isinstance(m, nn.Conv2d):
            torch.nn.init.xavier_uniform_(m.weight)
            torch.nn.init.zeros_(m.bias)

        self.features.apply(weights_init)
        torch.nn.utils.clip_grad_norm_(self.features.parameters(), max_norm=1)



    def forward(self, x):
        return self.features(x)


# Evaluate the model on test data
model = SimpleCNN()
model.load_state_dict(torch.load('layer_separation_model.pth'))
model.eval()
predictions_list = []
with torch.no_grad():
    for i in range(8):
      testing_tensor = torch.tensor(testers[i]).permute(2, 0, 1).to(torch.float32)
      test_loader = DataLoader(testing_tensor, batch_size=8, shuffle=False)
      for image in test_loader:
          image = image.unsqueeze(0)

          outputs = model(image)

          predictions = (outputs > 0.5).float()

          fig, axs = plt.subplots(1, 2, figsize=(15, 5))

          axs[0].imshow(cv2.cvtColor(testers[i], cv2.COLOR_BGR2RGB))
          axs[0].set_title("Original Image")

          prediction = np.array(predictions[0].squeeze())
          print(np.max(prediction))
          predictions_list.append(prediction)
          axs[1].imshow(prediction, cmap="gray")
          axs[1].set_title("Predicted Mask")

          plt.show()
