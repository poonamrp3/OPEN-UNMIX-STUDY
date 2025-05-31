# OPEN-UNMIX-STUDY
1. Performance benchmarking of Open-Unmix demonstrated that increasing batch size and hidden size does not always reduce loss or accelerate training.
2. The Open-Unmix model with a larger batch size of 16 and hidden size of 512 had a higher loss of 11.2, compared to Open-Unmix2 with a smaller batch size of 4 and hidden size of 256, which achieved a lower loss of 8.55.
3. Analysis indicated that a hidden size of 256 was optimal for processing 16KHz signals in the Open-Unmix model. Increasing hidden size beyond this point led to decreased performance due to overfitting, as the model began to capture very subtle frequency patterns.
