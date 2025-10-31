from typing import TypedDict
from sympy import N
# To generate a sturtured output

class Person (TypedDict):
    name: str
    age :int

new_person:Person={'name':'Jagan Pradhan', 'age':21}

print(new_person)
