import matplotlib.pyplot as plt

# Lists extracted from training log
epochs = list(range(1, 51))  # Epochs 1 to 50

training_loss = [
    0.4940, 0.2597, 0.1862, 0.1589, 0.1406, 0.1227, 0.1154, 0.1063, 0.0997, 0.0944,
    0.0904, 0.0851, 0.0848, 0.0818, 0.0764, 0.0814, 0.0738, 0.0700, 0.0729, 0.0709,
    0.0672, 0.0640, 0.0651, 0.0627, 0.0624, 0.0584, 0.0594, 0.0629, 0.0626, 0.0636,
    0.0556, 0.0589, 0.0552, 0.0549, 0.0579, 0.0550, 0.0568, 0.0567, 0.0513, 0.0550,
    0.0554, 0.0516, 0.0495, 0.0525, 0.0528, 0.0550, 0.0551, 0.0580, 0.0538, 0.0500
]

validation_loss = [
    0.3409, 0.2406, 0.2092, 0.1845, 0.1662, 0.1561, 0.1424, 0.1318, 0.1134, 0.1190,
    0.1168, 0.1050, 0.0977, 0.0986, 0.0978, 0.0871, 0.0803, 0.0826, 0.0854, 0.0793,
    0.0727, 0.0710, 0.0739, 0.0675, 0.0659, 0.0668, 0.0734, 0.0685, 0.0657, 0.0637,
    0.0601, 0.0661, 0.0614, 0.0586, 0.0602, 0.0557, 0.0603, 0.0510, 0.0527, 0.0554,
    0.0544, 0.0541, 0.0504, 0.0580, 0.0472, 0.0472, 0.0494, 0.0478, 0.0465, 0.0469
]

# Create the plot
plt.figure(figsize=(10, 6))
plt.plot(epochs, training_loss, label='Training Loss', marker='o', color='blue')
plt.plot(epochs, validation_loss, label='Validation Loss', marker='o', color='orange')

# Customize the plot
plt.title('Training and Validation Loss over Epochs', fontsize=16, fontweight='bold')
plt.xlabel('Epoch', fontsize=14)
plt.ylabel('Loss', fontsize=14)
plt.legend(fontsize=12)
plt.grid(True)
plt.tight_layout()

# Save the plot as 'epoch_plotter.png'
plt.savefig('epoch_plotter.png', dpi=300)
plt.show()
