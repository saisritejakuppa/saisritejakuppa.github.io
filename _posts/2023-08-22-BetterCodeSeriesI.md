---
layout: post
title:  Better Code Part I
date:   2023-08-22 16:40:16
description: 
tags: BetterCode
categories: BetterCode
---

# Good Names in Code

Names are meant to be meaningful, should understand what it contains without knowing whats happening behind the scenario.


This you will be naming often
1. variable/contants
    1. Contains dataset, results of a function... anything - use Nouns(userdata) or shortphrases with adjectives(isValid)
2. function/methods
    1. Use verbs(sendData) or short phrases with adjectives(isInputValid)
3. classes
    1. These are used to make an object - use Nouns or shortPhrases(User, RequestBody)


## Naming convections
1. SnakeCase - hello_world(python - variables, function/methods)
2. camelCase - helloWorld( java or script - variables, function/methods)
3. pascalCase - HelloWorld(python/java - Classes)


### Varaible, Constants and Properties
1. Values is an object.
    1. Describe the object 
        1. User
        2. database

    2. Provide More information of the variable
        1. Authentic User
        2. Sql Database

2. Value can be boolean.
    1. Tell True or False
        1. isValid
        2. loggedIn

    2. Provide More information of the variable
        1. isUserLoggedIN

3. Value can be number or string.
    1. Describe the value
        1. name
        2. age
    2. Provide More information of the variable
        1. firstname
        2. age

### Functions
1. Functions perform an operation
    1. Describe the operation
        1. GetUserInfo
        2. Response.send()
2. Computes a boolean
    1. Know about the status
        1. isValid()
        2. Purchase.isPaid()
Also you can add more information about to the name like EmailResponse.send(), this tells you are working on some email responding operation.

### Classes
1. Describe the Object
    1. UserProduct- it can become customer or course.


## Tips
1. Be consistent about the code names.( use get over fetch if you like it.)
2. Dont use bad language.
3. Dont overadd or under add the required info requried for the variable.



