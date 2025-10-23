import os
import glob
import tarfile
import cv2
import numpy as np
from tqdm import tqdm
from pathlib import Path


class ImageExtractor:

    def __init__(self, data_root="."):
        """Initialise avec le chemin racine des données"""
        self.data_root = Path(data_root)
        self.archive_file = self.data_root / "full_archive.tar.gz"
        self.extracted_dir = self.data_root / "extracted_images"
        self.npz_file = self.data_root / "dataset.npz"

    def _find_archive_parts(self):
        """Trouve les parties d'archive automatiquement"""
        parts = sorted(self.data_root.glob("curated.tar.gz.*"))
        if not parts:
            raise FileNotFoundError(
                f"Aucune partie d'archive trouvée dans {self.data_root}. "
                f"Placez curated.tar.gz.01, .02 etc. dans ce dossier."
            )
        return parts

    def reconstruct_archive(self):
        """Reconstitue l'archive complète si nécessaire"""
        if self.archive_file.exists():
            return True

        parts = self._find_archive_parts()
        print(f"Reconstitution de {self.archive_file} à partir de {len(parts)} parties...")

        with open(self.archive_file, 'wb') as outfile:
            for part in tqdm(parts, desc="Concaténation"):
                with open(part, 'rb') as infile:
                    outfile.write(infile.read())
        return True

    def _extract_single_image(self, member, tar):
        """Extrait et traite une seule image"""
        img_path = self.extracted_dir / member.name
        tar.extract(member, path=self.extracted_dir)

        img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
        if img is None:
            return None, None

        img = cv2.resize(img, (32, 32))
        label = os.path.basename(os.path.dirname(member.name))
        return img, label
    def extract_images(self):
        """Extrait toutes les images de l'archive"""
        if self.npz_file.exists():
            print(f"Chargement des données pré-existantes depuis {self.npz_file}")
            data = np.load(self.npz_file)
            return data['X'], data['y']

        print(f"Extraction des images vers {self.extracted_dir}...")
        os.makedirs(self.extracted_dir, exist_ok=True)

        X, y = [], []
        with tarfile.open(self.archive_file, "r:gz") as tar:
            members = [m for m in tar.getmembers()
                       if m.name.lower().endswith(('.png', '.jpg', '.jpeg'))]

            for member in tqdm(members, desc="Extraction"):
                try:
                    img, label = self._extract_single_image(member, tar)
                    if img is not None:
                        X.append(img)
                        y.append(label)
                except Exception as e:
                    print(f"Erreur sur {member.name}: {str(e)}")
                    continue

        X, y = np.array(X, dtype=np.uint8), np.array(y)
        np.savez(self.npz_file, X=X, y=y)
        print(f"Extraction terminée. {len(X)} images sauvegardées dans {self.npz_file}")
        return X, y

    def sortie(self, test_size=0.2, shuffle=True):
        """Retourne les données séparées en train/test"""
        X, y = self.extract_images()

        if shuffle:
            indices = np.random.permutation(len(X))
            X, y = X[indices], y[indices]

        split_idx = int(len(X) * (1 - test_size))
        return (
            X[:split_idx], y[:split_idx],  # train
            X[split_idx:], y[split_idx:]  # test
        )



if __name__ == "__main__":
    # Exemple d'utilisation directe
    extractor = ImageExtractor()
    extractor.reconstruct_archive()
    x_train, y_train, x_test, y_test = extractor.sortie()
    print(f"Train: {len(x_train)} images, Test: {len(x_test)} images")
    print(y_train.shape)