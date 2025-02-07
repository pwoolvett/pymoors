# Genetic Operators

In pymoors, the genetic operators expected by the various algorithms are **mutation**, **crossover**, and a **sampler**. The selection and tournament operators are usually built into the algorithms; for example, NSGA2 employs selection and tournament procedures based on Rank and Crowding.

Currently, pymoors comes with a battery of **pre-defined genetic operators** implemented in Rust. The goal is to continuously add more classic genetic operators to reduce the amount of Python code that needs to be executed.

!!! warning 
    At the moment, pymoors does not provide a way to define custom genetic operators using NumPy functions. This feature is planned to be available as soon as possible.

Each genetic operator in pymoors is exposed to Python as a class. For example, consider the following:

```python
from pymoors import RandomFloatSampling, GaussianMutation, ExponentialCrossover,

# Create a sampler that generates individuals randomly between 0 and 10.
sampler = RandomFloatSampling(min=0, max=10)
# Create a gauss mutator instance
mutation=GaussianMutation(gene_mutation_rate=0.1, sigma=0.01)
# Create an exponential crossover instance
crossover=ExponentialCrossover(exponential_crossover_rate = 0.75)
```

!!! warning 
    1. Currently, these instances serve only as a means to inform Rust which operator to use, but they do not expose any public methods to Python; everything functions internally within the Rust core. We are currently evaluating whether it is necessary to expose these methods to Python so that users can interact with them.
    2. This section will be updated with more information on all the genetic operators in the near future.
