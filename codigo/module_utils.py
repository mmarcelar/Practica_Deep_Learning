import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torch.nn.functional as F
import matplotlib.pyplot as plt
import torchvision
import random
import requests
from PIL import Image
from io import BytesIO

#############################
# Funciones de visualización#
#############################

def show_augmented_images(trainloader, mean, std, transforms_list=None):
    """
    Muestra imágenes con diferentes técnicas de data augmentation aplicadas.
    Args:
        trainloader: DataLoader con las imágenes
        transforms_list: Lista de tuplas (nombre_transformacion, transformacion). Si es None,
                        se usará una lista predeterminada de transformaciones.
    """
    if transforms_list is None:
        transforms_list = [
            ('Horizontal Flip', torchvision.transforms.RandomHorizontalFlip(p=1.0)),
            ('Rotation 30°', torchvision.transforms.RandomRotation(degrees=30)),
            ('Color Jitter', torchvision.transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.5)),
            ('Random Crop', torchvision.transforms.RandomCrop(size=32, padding=4))
        ]

    # Obtener una imagen del trainloader
    images, _ = next(iter(trainloader))
    img = images[0]  # Tomamos la primera imagen del batch
    # Deshacer la normalización para visualización
    if mean is not None and std is not None:
        # Desnormalizar usando mean y std proporcionados
        img = img * std + mean
    # Crear una figura
    num_transforms = len(transforms_list)
    fig = plt.figure(figsize=(15, 3 * ((num_transforms + 4) // 5)))
    
    # Mostrar imagen original
    ax = fig.add_subplot(((num_transforms + 4) // 5), 5, 1, xticks=[], yticks=[])
    img_show = img.numpy().transpose((1, 2, 0))
    ax.imshow(img_show)
    ax.set_title('Original')
    
    # Aplicar y mostrar cada transformación
    for idx, (name, transform) in enumerate(transforms_list, 1):
        ax = fig.add_subplot(((num_transforms + 4) // 5), 5, idx + 1, xticks=[], yticks=[])
        
        # Aplicar transformación
        augmented = transform(img)
        
        # Convertir a numpy y transponer para visualización
        if isinstance(augmented, torch.Tensor):
            augmented = augmented.numpy().transpose((1, 2, 0))
            
        ax.imshow(augmented)
        ax.set_title(name)
    
    plt.tight_layout()
    plt.show()

def show_random_images(trainloader, class_names, mean=None, std=None):
    # Obtener un batch de imágenes del trainloader
    images, labels = next(iter(trainloader))
    
    # Crear una figura con subplots
    fig = plt.figure(figsize=(10, 4))
    
    # Para cada clase, mostrar una imagen aleatoria
    for i in range(len(class_names)):
        ax = fig.add_subplot(2, 5, i + 1, xticks=[], yticks=[])
        
        # Encontrar todas las imágenes de la clase actual
        idx = (labels == i).nonzero(as_tuple=True)[0]
        if len(idx) > 0:
            # Seleccionar una imagen aleatoria de esta clase
            random_idx = random.choice(idx)
            img = images[random_idx]
        else:
            continue
            
        # Deshacer la normalización para visualización
        if mean is not None and std is not None:
            # Desnormalizar usando mean y std proporcionados
            img = img * std + mean

        # Transponer la imagen para que matplotlib pueda mostrarla correctamente
        img = img.numpy().transpose((1, 2, 0))
        
        ax.imshow(img)
        ax.set_title(class_names[i])
    
    plt.tight_layout()
    plt.show()

def plot_training_curves(train_losses, val_losses, train_accs, val_accs, num_epochs, test_acc=None):
    plt.style.use("ggplot")
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(range(num_epochs), train_losses, label="Train Loss")
    plt.plot(range(num_epochs), val_losses, label="Validation Loss")
    plt.title("Training and Validation Loss")
    plt.xlabel("Epoch #")
    plt.ylabel("Loss")
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(range(num_epochs), train_accs, label="Train Accuracy")
    plt.plot(range(num_epochs), val_accs, label="Validation Accuracy")
    if test_acc is not None:
        plt.axhline(y=test_acc, color='red', linestyle='--', label='Test Accuracy')
    plt.title("Training and Validation Accuracy")
    plt.xlabel("Epoch #")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.tight_layout()
    plt.show()


#############################
# Funciones de entrenamiento#
#############################


# download utils
def download_and_display_image(url: str) -> Image.Image:
    """
    Descarga y muestra una imagen desde una URL.
    
    Args:
        url: URL de la imagen a descargar
        
    Returns:
        Image: Objeto de imagen PIL
    """
    try:
        # Download and display the image
        headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'}
        response = requests.get(url, headers=headers)
        response.raise_for_status()  # Raise an exception for bad status codes
        
        image = Image.open(BytesIO(response.content))
        # Convert to RGB if image is in RGBA mode
        if image.mode == 'RGBA':
            image = image.convert('RGB')
            
        plt.imshow(image)
        plt.axis('off')
        plt.show()
        return image
    except Exception as e:
        print(f"Error downloading image: {str(e)}")
        return None
    
# Definimos la función para entrenar una época (sacada del notebook II)
def train_epoch(model: nn.Module, device: torch.device, train_loader: DataLoader, 
                criterion, optimizer, l1_lambda=None, scheduler=None):
    """
    Entrena una época de la red neuronal y devuelve las métricas de entrenamiento.
    
    Args:
        model: Modelo de red neuronal a entrenar
        device: Dispositivo donde se realizará el entrenamiento (CPU/GPU)
        train_loader: DataLoader con los datos de entrenamiento
        criterion: Función de pérdida a utilizar
        optimizer: Optimizador para actualizar los pesos
        scheduler: Scheduler para ajustar el learning rate
        
    Returns:
        train_loss: Pérdida promedio en el conjunto de entrenamiento
        train_acc: Precisión en el conjunto de entrenamiento (%)
        current_lr: Learning rate actual después del scheduler
    """
    model.train()
    train_loss = 0
    correct = 0
    total = 0

    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        if l1_lambda is not None:
            l1_norm = sum(p.abs().sum() for p in model.parameters())
            loss += l1_lambda * l1_norm
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
        _, predicted = output.max(1)
        total += target.size(0)
        correct += predicted.eq(target).sum().item()

    train_loss /= len(train_loader)
    train_acc = 100. * correct / total

    # Aplicar el scheduler después de cada época
    if scheduler is not None:
        scheduler.step()
        current_lr = scheduler.get_last_lr()[0]
        return train_loss, train_acc, current_lr
    else:
        return train_loss, train_acc


def eval_epoch(model: nn.Module, device: torch.device, val_loader: DataLoader, 
               criterion):
    """
    Evalúa el modelo en el conjunto de validación.
    
    Args:
        model: Modelo de red neuronal a evaluar
        device: Dispositivo donde se realizará la evaluación (CPU/GPU)
        val_loader: DataLoader con los datos de validación
        criterion: Función de pérdida a utilizar
        
    Returns:
        val_loss: Pérdida promedio en el conjunto de validación
        val_acc: Precisión en el conjunto de validación (%)
    """
    model.eval()
    val_loss = 0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for data, target in val_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            loss = criterion(output, target)
            val_loss += loss.item()
            _, predicted = output.max(1)
            total += target.size(0)
            correct += predicted.eq(target).sum().item()

    val_loss /= len(val_loader)
    val_acc = 100. * correct / total

    return val_loss, val_acc

def evaluate_model(model, testloader, device):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for data in testloader:
            images, labels = data[0].to(device), data[1].to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    return 100. * correct / total

def plot_training_curves(train_losses, val_losses, train_accs, val_accs, num_epochs, test_acc=None):
    plt.style.use("ggplot")
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(range(num_epochs), train_losses, label="Train Loss")
    plt.plot(range(num_epochs), val_losses, label="Validation Loss")
    plt.title("Training and Validation Loss")
    plt.xlabel("Epoch #")
    plt.ylabel("Loss")
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(range(num_epochs), train_accs, label="Train Accuracy")
    plt.plot(range(num_epochs), val_accs, label="Validation Accuracy")
    if test_acc is not None:
        plt.axhline(y=test_acc, color='red', linestyle='--', label='Test Accuracy')
    plt.title("Training and Validation Accuracy")
    plt.xlabel("Epoch #")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.tight_layout()
    plt.show()