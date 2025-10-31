from openai import files
from pydantic import BaseModel,EmailStr,Field
from typing import Optional
class Student(BaseModel):
    name:str='jagan pradhan' # setting default value as jagan pradhan
    age:Optional[int]=None # optional feild if nothing is passed it None or Null
    email:EmailStr
    cgpa:float=Field(gt=0,lt=10,default=0,description="It contains the float value which represents the vaalue")
    mobile: str = Field(
        ..., 
        pattern=r'^[6-9]\d{9}$', 
        description="Valid Indian phone number"
    )


new_student = { 'email': 'jp515@gmail.com','cgpa':6.1,'mobile':'9876543210'}

student =Student(**new_student)
432
#print(student.name)

dict_student=dict(student)

json_student=student.model_dump_json()
print(student)
print(json_student)