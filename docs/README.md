# Documents

This folder aims at becoming the place where our notes, write-ups and thoughts are gathered in the form of LaTeX, Markdown, or whatever other format, documents. For this reason, we propose the following base folder structure:

* [notes](notes/) Subfolder where to save notes with derivations or other content which would be beneficial for any person contributing to the repository
* [bib](bib/): Do not upload PDF files, but the reference file (`.bib`) 
* [how-to](howto): Subfolder where guides on how to do technical things will be placed. The underlying idea is that, if something took you more than 30 minutes to figure it out, it'll probably take the same amount of time to others (or even more). Hence, it'll be great if we can share with each other what we struggle with, and how we solved it. 

## Bibliography 

This space should be devoted to keep a nice collection of references, books, pre-prints or websites which are useful for anyone to check and learn. The idea is that we keep alive the list of references, and interconnected with the bib-items in the [bib folder](docs/bib). When you include a new item, please place the corresponding `.bib` file in such a folder, and link the item with it. We'll usually add some short introduction here about the reference so that everyone can know what is and is not worth their attention. As per usual, follow the style rules you obvserve from the already existing items.

### Pre-prints

- [Neural ODEs](https://arxiv.org/pdf/1806.07366v5.pdf): This is the main paper on neural ODEs where they were introduced by Chen *et al.* 

### References

(TBD)

### Websites

- [Depth First](https://www.depthfirstlearning.com/): DFL is a compendium of curricula to help you deeply understand Machine Learning.
- [Neural ODEs course](https://www.depthfirstlearning.com/2019/NeuralODEs#:~:text=Neural%20ODEs%20are%20neural%20network,efficiently%20train%20models%20via%20ODEs.): This is a course (set of notes) to understand the main paper about neural ODES ([this](https://arxiv.org/abs/1806.07366)). It goes from the very basics on ODEs to the discussion on the adjoint method and auto-diff.
- [Easy neural ODEs](https://github.com/jacobjinkelly/easy-neural-ode): The repository where we can find the code of the paper [learning differential equations that are easy to solve](https://arxiv.org/pdf/2007.04504.pdf).
- [Augmented neural ODES](https://github.com/EmilienDupont/augmented-neural-odes): This is the repository where we can find the code of the paper on [augmented neural ODEs](https://arxiv.org/abs/1904.01681). In here we can find the actual implementation of (augmented) neural ODEs. 
