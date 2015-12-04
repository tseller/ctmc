ctmc
====

With the `ctmc` module, we can represent Continuous-Time Markov Chains and propagate them over time.

There is a base class `CTMC` that handles all the matrix math. On top of this, we provide a subclass `BirthDeath` which represents a popular example of a Markov Chain: a "birth-death" process (https://en.wikipedia.org/wiki/Birth%E2%80%93death_process).

We also provide a subclass `SupplyDemand` which could represent customers (demand) arriving at a bank with a fixed number of tellers (supply), and with the added twist that customers might linger before deciding to approach an already-available teller (and actually might decide to leave rather than see a teller at all).

Feel free to add more subclasses.

#Install
```python
pip install git+https://www.github.com/tseller/ctmc.git
```

#Usage
Here's an example with BirthDeath. Suppose units are "born" at an average rate 1, and die at an average rate 1 (per existing unit, so the death rate is proportional to the current population). Let's set a cap of 9 possible units (so 10 states, 0-9), meaning the birth rate drops to zero whenever the population is 9.

```python
bd = BirthDeath(10, 1, 1)
```

In the population is 5 at t=0, the expected population at t=10 is
```python
bd.propagate(metric='population', state=5, t=10)
4.737...
```
At that time, the rate of change of expected population is 
```python
bd.differentiate(metric='population', state=5, t=2)
-0.023...
```
At that time, the distribution of possible populations is
```python
bd.distribution(metric='population', t=10, state=5)
[(0, 0.084442394986868882),
 (1, 0.087076931647434005),
 (2, 0.091670505965224872),
 (3, 0.097091892337860058),
 (4, 0.10211844327017415),
 (5, 0.10581806731685438),
 (6, 0.10781288023639653),
 (7, 0.1083291717680195),
 (8, 0.10801809660650136),
 (9, 0.1076216158646677)]
 ```
#TODO
Add the ability to find equilbrium states.

