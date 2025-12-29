import numpy as np
import matplotlib.pyplot as plt
from rdkit import DataStructs
from rdkit.DataStructs.cDataStructs import CreateFromBitString

def array_to_explicit_bitvect(array):
    """
    Convert a numpy.ndarray (binary, length 2048) to RDKit ExplicitBitVect.
    """
    if not isinstance(array, np.ndarray) or len(array) != 2048:
        raise ValueError("Each fingerprint must be a numpy.ndarray of length 2048.")
    bitstring = ''.join(map(str, array.astype(int)))
    return CreateFromBitString(bitstring)

def calculate_and_plot_similarity(fingerprint_list, save_path):
    """
    Calculate Tanimoto similarity for a list of binary 2048-length arrays
    relative to the last array, and save the similarity plot.

    Parameters:
    - fingerprint_list: list of 2048-length numpy.ndarrays (binary)
    - save_path: file path to save the resulting plot
    """
    if not fingerprint_list or not isinstance(fingerprint_list[-1], np.ndarray):
        raise ValueError("Fingerprint list must have a valid last fingerprint as a numpy.ndarray.")

    # Convert numpy arrays to RDKit ExplicitBitVect
    rdkit_fps = [
        array_to_explicit_bitvect(fp) if fp is not None else None
        for fp in fingerprint_list
    ]

    # Reference fingerprint (last element in the list)
    reference_fp = rdkit_fps[-1]
    if reference_fp is None:
        raise ValueError("The last fingerprint cannot be None.")

    # Calculate similarities
    similarities = []
    for idx, fp in enumerate(rdkit_fps):
        if fp is None:
            similarity = 0
        else:
            similarity = DataStructs.TanimotoSimilarity(fp, reference_fp)
        similarities.append(similarity)

    # Plot similarities
    plt.figure(figsize=(10, 6))
    plt.plot(range(len(similarities)), similarities, marker='o', linestyle='-', label='Similarity')
    plt.title("Fingerprint Similarity to the Last Element")
    plt.xlabel("Index")
    plt.ylabel("Tanimoto Similarity")
    plt.ylim(0, 1.1)
    plt.grid(True)
    plt.legend()
    plt.tight_layout()

    # Save the plot
    plt.savefig(save_path)
    plt.close()
    print(f"Plot saved at {save_path}")

# Example Usage
if __name__ == "__main__":
    # Create a list of example fingerprints as 2048-length numpy arrays
    test_fingerprints = [np.random.randint(0, 2, 2048) for _ in range(10)]
    test_fingerprints.append(None)  # Add a None to simulate missing fingerprint

    # Save the plot to a specific location
    output_path = "/root/targetdiff/xxd_test_data/figs/test1.png"
    calculate_and_plot_similarity(test_fingerprints, output_path)