# Typographic Style Transfer

This is the code repository of my master project. I propose a style transfer approach for the generation of letters in the style of a given font.

In the first step I train a deep convolutional neural network to determine whether two glyphs belong to the same font or not. The models’s internal representation is then used in the style transfer task. Based on the classifier’s response an input letter is modified such that its similarity to the target font is increased.

The results show that the presented method is solely generating adversarial examples, and it is ultimately deemed unsuccessful.

## Data Set

[Multi-Content GAN for Few-Shot Font Style Transfer](https://github.com/azadis/MC-GAN); [Samaneh Azadi](https://people.eecs.berkeley.edu/~sazadi/), [Matthew Fisher](https://research.adobe.com/person/matt-fisher/), [Vladimir Kim](http://vovakim.com/), [Zhaowen Wang](https://research.adobe.com/person/zhaowen-wang/), [Eli Shechtman](https://research.adobe.com/person/eli-shechtman/), [Trevor Darrell](https://people.eecs.berkeley.edu/~trevor/), in arXiv, 2017.

