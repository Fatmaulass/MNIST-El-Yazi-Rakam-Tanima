import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
import random
import numpy as np

def set_seed(seed=20):
    """
    Sonuçların tekrarlanabilir olması için rastgelelik içeren 
    tüm kütüphanelerin çekirdek (seed) değerini sabitler.
    """
    random.seed(seed)               # Python'ın kendi random kütüphanesi
    np.random.seed(seed)            # Numpy kütüphanesi
    torch.manual_seed(seed)         # PyTorch (CPU)
    
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)        # PyTorch (GPU - Tekli)
        torch.cuda.manual_seed_all(seed)    # PyTorch (GPU - Çoklu)
        
        # GPU optimizasyonlarını deterministik moda zorlar
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

set_seed(42)


# 1. AYARLAR
batch_size = 64
learning_rate = 0.001    
epochs = 7
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 2. VERİ SETİ - NORMALİZASYON
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,)) 
])

train_data = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
test_data  = datasets.MNIST(root='./data', train=False, download=True, transform=transform)

train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True) 
test_loader  = torch.utils.data.DataLoader(test_data,  batch_size=1000, shuffle=False)

# 3. MODEL (TEK GİZLİ KATMAN)
class SimpleMLP(nn.Module):
    def __init__(self):
        super(SimpleMLP, self).__init__()       
        self.fc1 = nn.Linear(784, 512)  # GİRİŞ -> GİZLİ KATMAN (784 -> 512)
        self.dropout = nn.Dropout(p=0.25)  # Dropout nöronların %25i hariç
        self.fc2 = nn.Linear(512, 10) # GİZLİ KATMAN -> ÇIKIŞ (512 -> 10)
       
    def forward(self, x):
        x = x.view(-1, 28 * 28) 
             
        x = torch.relu(self.fc1(x)) # 1. Katman fc1 işlemleri
        x = self.dropout(x) 

        x = self.fc2(x) # 2. Katman (Çıkış) işlemleri   
        return x
        # arada başka katman yok, direkt fc2'ye veriyor

model = SimpleMLP().to(device)
criterion = nn.CrossEntropyLoss() #Loss Function
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# --- SCHEDULER ---
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)

# Tarihçe listeleri
history = {'train_loss': [], 'test_loss': [], 'train_acc': [], 'test_acc': []}

print(f"Eğitim Başladı ({device}) - Tek Katmanlı Model...")



# 4. EĞİTİM DÖNGÜSÜ
for epoch in range(epochs):
    # --- TRAIN AŞAMASI ---
    model.train()
    train_loss = 0
    correct_train = 0
    total_train = 0
    
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        train_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total_train += labels.size(0)
        correct_train += (predicted == labels).sum().item()
    
    avg_train_loss = train_loss / len(train_loader)
    avg_train_acc = 100 * correct_train / total_train


    

    # --- TEST AŞAMASI ---
    model.eval()
    test_loss = 0
    correct_test = 0
    total_test = 0
    
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            test_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total_test += labels.size(0)
            correct_test += (predicted == labels).sum().item()
    
    avg_test_loss = test_loss / len(test_loader)
    avg_test_acc = 100 * correct_test / total_test

    # Scheduler güncelle
    scheduler.step()


    current_lr = scheduler.get_last_lr()[0] 
    history['train_loss'].append(avg_train_loss)
    history['test_loss'].append(avg_test_loss)
    history['train_acc'].append(avg_train_acc)
    history['test_acc'].append(avg_test_acc)
    print(f"Epoch {epoch+1}/{epochs} | LR: {current_lr:.5f} | Train Acc: %{avg_train_acc:.2f} | Test Acc: %{avg_test_acc:.2f} | Train Loss: {avg_train_loss:.4f}")

# 5. DETAYLI ANALİZ
print("\n--- Detaylı Hata Analizi ---")
nb_classes = 10
confusion_matrix_res = torch.zeros(nb_classes, nb_classes)

with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        _, preds = torch.max(outputs, 1)
        for t, p in zip(labels.view(-1), preds.view(-1)):
            confusion_matrix_res[t.long(), p.long()] += 1

# Doğruluk oranları
for i in range(nb_classes):
    class_acc = 100 * confusion_matrix_res[i, i] / confusion_matrix_res[i].sum()
    print(f"Rakam {i} için doğruluk: %{class_acc:.2f}")

# En çok karıştırılanlar
print("\n--- En Çok Yapılan Hatalar ---")
errors = confusion_matrix_res.clone()
for i in range(nb_classes):
    errors[i, i] = 0

max_error_val = errors.max()
most_confused_indices = (errors == max_error_val).nonzero(as_tuple=False)

for idx in most_confused_indices:
    gercek = idx[0].item()
    tahmin = idx[1].item()
    hata_sayisi = errors[gercek, tahmin].item()
    print(f"DİKKAT: Model {gercek} rakamını {hata_sayisi} kez '{tahmin}' sanarak karıştırdı!")

# 6. GRAFİKLEŞTİRME
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(history['train_loss'], label='Eğitim Kaybı')
plt.plot(history['test_loss'], label='Test Kaybı', linestyle='--')
plt.title("Hata/Kayıp Grafiği")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history['train_acc'], label='Eğitim Doğruluğu')
plt.plot(history['test_acc'], label='Test Doğruluğu', linestyle='--')
plt.title("Doğruluk Grafiği")
plt.xlabel("Epoch")
plt.ylabel("Accuracy (%)")
plt.legend()

plt.tight_layout()
# plt.show()

# 7. HEATMAP
plt.figure(figsize=(10, 8))
cm_numpy = confusion_matrix_res.cpu().numpy().astype(int)
sns.heatmap(cm_numpy, annot=True, fmt='d', cmap='Blues', cbar=False,
            xticklabels=range(10), yticklabels=range(10))
plt.xlabel("Tahmin Edilen")
plt.ylabel("Gerçek Değer")
plt.title("Confusion Matrix")
plt.show()

# 8. MODELİ KAYDETME
save_path = "mnist_mlp_model.pth"
torch.save(model.state_dict(), save_path)

print(f"\nModelin ağırlıkları '{save_path}' dosyasına kaydedildi")

