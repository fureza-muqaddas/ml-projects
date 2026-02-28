
from setuptools import find_packages, setup
from typing import List

def get_requirements(file_path:str)->List[str]:
    '''

    this function will return a list of requirments 
    
    '''
    requirements=[]
    with open(file_path) as file_obj:
        requirements = file_obj.readlines()
        requirements = [req.replace("\n", "") for req in requirements]
    return requirements

setup(
   name='mlproject',
   version='1.0',
   author='Fureza Muqaddas',
   author_email='furezamuqaddas@gmail.com',
   packages= find_packages(),  #same as name
   install_requires= get_requirements('requirements.txt') #external packages as dependencies
)
