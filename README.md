# Ride Hitch
Use DRL method to help solve the ridehitch problem.
## Problem Setting
There are two types of requests from the environment. 
One type is the supply type from drivers and another is the demand from passengers.


## Result
|                     | FIRST | RANDOM | DQN | OPT  |
|---------------------|-------|--------|-----|------|
| norm100T50D50P100   | 21    | 23     |     | 24   |
| norm1000T50D50P100  | 353   | 334    |     | 379  |
| norm10000T50D50P100 | 3588  | 3323   |     | 3779 |
