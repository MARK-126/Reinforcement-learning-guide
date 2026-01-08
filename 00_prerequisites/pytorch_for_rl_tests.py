"""
Tests para pytorch_for_rl.ipynb
"""
import torch
import torch.nn as nn


def crear_tensor_test(crear_tensor):
    """Test para ejercicio 1: crear_tensor"""
    print("Testing crear_tensor...")
    
    # Test 1: Tipo correcto
    tensor = crear_tensor([1.0, 2.0, 3.0])
    assert isinstance(tensor, torch.Tensor), "❌ Debe retornar torch.Tensor"
    
    # Test 2: Valores correctos
    expected = torch.tensor([1.0, 2.0, 3.0])
    assert torch.allclose(tensor, expected), "❌ Valores incorrectos"
    
    # Test 3: Tipo float
    assert tensor.dtype == torch.float32, "❌ Debe ser float32"
    
    print("✅ Test pasado! Creas tensores correctamente.")
    return True


def crear_red_test(crear_red):
    """Test para ejercicio 2: crear_red"""
    print("Testing crear_red...")
    
    # Test 1: Retorna nn.Module
    red = crear_red(4, 8, 2)
    assert isinstance(red, nn.Module), "❌ Debe retornar nn.Module"
    
    # Test 2: Tiene las capas correctas
    assert hasattr(red, 'fc1'), "❌ Debe tener capa fc1"
    assert hasattr(red, 'fc2'), "❌ Debe tener capa fc2"
    
    # Test 3: Dimensiones correctas
    assert red.fc1.in_features == 4, "❌ fc1 input incorrecto"
    assert red.fc1.out_features == 8, "❌ fc1 output incorrecto"
    assert red.fc2.in_features == 8, "❌ fc2 input incorrecto"
    assert red.fc2.out_features == 2, "❌ fc2 output incorrecto"
    
    print("✅ Test pasado! Red creada correctamente.")
    return True


def forward_pass_test(forward_pass, crear_red):
    """Test para ejercicio 3: forward_pass"""
    print("Testing forward_pass...")
    
    red = crear_red(4, 8, 2)
    input_tensor = torch.tensor([1.0, 2.0, 3.0, 4.0])
    
    # Test 1: Retorna tensor
    output = forward_pass(red, input_tensor)
    assert isinstance(output, torch.Tensor), "❌ Debe retornar tensor"
    
    # Test 2: Shape correcto
    assert output.shape == torch.Size([2]), f"❌ Shape incorrecto: {output.shape}"
    
    # Test 3: Valores numéricos
    assert not torch.isnan(output).any(), "❌ Output contiene NaN"
    
    print("✅ Test pasado! Forward pass funciona.")
    return True


def calcular_loss_test(calcular_loss):
    """Test para ejercicio 4: calcular_loss"""
    print("Testing calcular_loss...")
    
    pred = torch.tensor([1.0, 2.0])
    target = torch.tensor([1.5, 2.5])
    
    # Test 1: Retorna tensor
    loss = calcular_loss(pred, target)
    assert isinstance(loss, torch.Tensor), "❌ Debe retornar tensor"
    
    # Test 2: Loss correcto (MSE)
    expected_loss = ((1.0-1.5)**2 + (2.0-2.5)**2) / 2
    assert abs(loss.item() - expected_loss) < 0.001, \
        f"❌ Loss incorrecto. Esperado: {expected_loss}, Obtenido: {loss.item()}"
    
    # Test 3: Loss es escalar
    assert loss.dim() == 0, "❌ Loss debe ser escalar"
    
    print("✅ Test pasado! Loss calculado correctamente.")
    return True


def entrenar_paso_test(entrenar_paso, crear_red):
    """Test para ejercicio 5: entrenar_paso"""
    print("Testing entrenar_paso...")
    
    red = crear_red(4, 8, 2)
    optimizer = torch.optim.Adam(red.parameters(), lr=0.01)
    
    input_tensor = torch.tensor([1.0, 2.0, 3.0, 4.0])
    target_tensor = torch.tensor([0.5, 1.5])
    
    # Guardar pesos iniciales
    initial_weights = red.fc1.weight.data.clone()
    
    # Test 1: Retorna loss
    loss = entrenar_paso(red, optimizer, input_tensor, target_tensor)
    assert isinstance(loss, torch.Tensor), "❌ Debe retornar loss"
    
    # Test 2: Pesos cambiaron
    assert not torch.equal(initial_weights, red.fc1.weight.data), \
        "❌ Los pesos deben cambiar después del entrenamiento"
    
    # Test 3: Loss disminuye con múltiples pasos
    losses = []
    for _ in range(10):
        loss = entrenar_paso(red, optimizer, input_tensor, target_tensor)
        losses.append(loss.item())
    
    assert losses[-1] < losses[0], \
        "❌ Loss debería disminuir con el entrenamiento"
    
    print("✅ Test pasado! Entrenamiento funciona correctamente.")
    print(f"   Loss inicial: {losses[0]:.4f}, Loss final: {losses[-1]:.4f}")
    return True


if __name__ == "__main__":
    print("Tests para pytorch_for_rl.ipynb")
    print("=" * 50)
