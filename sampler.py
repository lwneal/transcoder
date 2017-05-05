import numpy as np


# Actually boltzmann(log(x)) for stability
def boltzmann(pdf, temperature=1.0, epsilon=1e-5):
    if temperature < epsilon:
        return pdf / (pdf.sum() + epsilon)
    pdf = np.log(pdf) / temperature
    x = np.exp(pdf)
    sums = np.sum(x, axis=-1)[:, np.newaxis] + epsilon
    return x / sums


def sample(pdfs):
    max_words, vocab_size = pdfs.shape
    samples = np.zeros(max_words)
    for i in range(len(samples)):
        samples[i] = np.random.choice(np.arange(vocab_size), p=pdfs[i])
    return samples
