# Readme  
  
use following step to using this project\  
this project using **django** and **python 3.x**.\  
**python 3.10** is recommended  
  
## First initialize project  
1. Clone project from github directory 
   `git clone https://github.com/rsakml/onlinecompailer.git`
   
2. Create a virtual environment in the project folder
   `python -m venv env`
  
3. Install in env folder
   `pip install -r requirements.txt`

   all project dependency is located on `requirements.txt`  

4. If Django is not installed, run the following command
   `pip install django`


Running Django Framework using Virtual Environment use this command:  
`onlinecompiler> env\Scripts\activate`

Virtual environment will be activate on command prompt if look like this:  
`(env) onlinecompiler>`

And run the python: 
`(env) onlinecompiler>python manage.py runserver `
it will serve on default port 8000.  
    
    
## API DOC  
  
| URL                    | Description               | method | params | response |  
|------------------------|---------------------------|--------|--------|----------|  
|  `/compile/run` | Compile java file | POST | ```{code:'xxx', user: 'you@mail.com'}```|```{output: { java: 'xx', test_output: 'xx', point: x}}```                 
| `/compile/test_files`|Get all java Test Files | GET ||
|`/compile/upload`|Upload java test file|POST|`{file:'xx'(multipart/form-data)}`|`{status: "ok"}`
|`compile/delete`|Delete java test file|POST|`{filename: 'xx'}`| ```{"message": "readme.txt deleted","status": "success"}```
