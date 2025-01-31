import os

valid_dir = "C:\\Users\\juan.ochoa\\clasificaTatuaje\\Clasificar tattoo\\valid_org"

if os.path.exists(valid_dir):
    print("✅ La ruta de validación existe.")
else:
    print("❌ ERROR: La ruta de validación NO existe. Verifica la ubicación.")