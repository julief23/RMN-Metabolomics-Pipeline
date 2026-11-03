import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import os
import math

# Chemin vers les courbes de calibration
input_dir = "results/calibration"

def plot_grid(curve_type, output_filename):
    """
    Génère une grille d'images pour un type de courbe donné.
    curve_type : 'pure_shift' ou 'standard'
    """
    output_path = os.path.join(input_dir, output_filename)

    # Collecte des images
    images_paths = []
    for compound in sorted(os.listdir(input_dir)):
        path = os.path.join(input_dir, compound, f"calibration_{compound}_{curve_type}.png")
        if os.path.exists(path):
            images_paths.append((compound, path))

    if not images_paths:
        print(f"Aucune image trouvée pour le type : {curve_type}")
        return

    # Création de la grille
    n = len(images_paths)
    ncols = 3
    nrows = math.ceil(n / ncols)

    fig, axes = plt.subplots(nrows, ncols, figsize=(15, 5 * nrows))
    axes = axes.flatten()

    for i, (compound, path) in enumerate(images_paths):
        img = mpimg.imread(path)
        axes[i].imshow(img)
        axes[i].axis("off")
        axes[i].set_title(compound, fontsize=10)

    # Masquer les cases vides
    for j in range(i + 1, len(axes)):
        axes[j].axis("off")

    plt.suptitle(f"Courbes {curve_type.replace('_', ' ').title()}", fontsize=14, y=1.01)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"Figure sauvegardée : {output_path}")


# Génération des deux grilles
plot_grid("pure_shift", "all_pure_shift_grid.png")
plot_grid("standard", "all_standard_grid.png")