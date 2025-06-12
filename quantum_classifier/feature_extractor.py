import os
import pandas as pd
import numpy as np
from PIL import Image
import torch
from torchvision import models, transforms
from sklearn.decomposition import PCA

# 1. Load CSV mapping image name to label
meta = pd.read_csv('masterfile.csv', header=None, names=['image', 'iga_class'])

# 2. Set up CNN for feature extraction (using ResNet18, no classifier)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = models.resnet18(pretrained=True)
model = torch.nn.Sequential(*list(model.children())[:-1])  # Remove classification layer
model.eval().to(device)
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

# 3. Extract CNN features for all images in the CSV
img_dir = 'data'  # <-- update this to your folder path
feature_list = []
valid_image_names = []

for idx, row in meta.iterrows():
    img_path = os.path.join(img_dir, row['image'])
    if not os.path.exists(img_path):
        print(f"Warning: {img_path} not found, skipping.")
        continue
    img = Image.open(img_path).convert('RGB')
    tensor = transform(img).unsqueeze(0).to(device)
    with torch.no_grad():
        feats = model(tensor)
    feature_list.append(feats.cpu().numpy().flatten())
    valid_image_names.append(row['image_name'])

# 4. Dimensionality reduction to N features (e.g., 10)
N_FEATURES = 10
all_feats = np.array(feature_list)
pca = PCA(n_components=N_FEATURES)
reduced_feats = pca.fit_transform(all_feats)

# 5. Prepare final DataFrame: filename, f1, f2, ..., fN, iga_class
# (keep only labels for images actually found)
final_meta = meta[meta['image_name'].isin(valid_image_names)].reset_index(drop=True)
df_out = pd.DataFrame(reduced_feats, columns=[f'feat_{i+1}' for i in range(N_FEATURES)])
df_out.insert(0, 'image_name', valid_image_names)
df_out['iga_class'] = final_meta['iga_class']

# 6. Save to CSV
df_out.to_csv('quantum_features_with_labels.csv', index=False)
print("Saved features and IGA labels to quantum_features_with_labels.csv")
