#!/usr/bin/env python
# coding: utf-8

# In[39]:


#!/usr/bin/env python
# coding: utf-8

# In[15]:


# -*- coding: utf-8 -*-
import re
import collections
import os
import sys
def get_file_content(path):
    bag_of_text = {}
    count = 0
    with open(path) as fp:
        for line in fp:
            bag_of_text[count] = line.strip()
            count+=1
    return bag_of_text

def get_complet_path(path):
    program_dir = os.path.dirname('__file__')
    return os.path.join(program_dir, path)
ingredients_fn = "./data/ingredients.txt"
solutions_fn="./data/ingredients_solutions.txt"

# Mettre dans cette partie la (les) expression(s) régulière(s)
# que vous utilisez pour analyser les ingrédients
#
# Vos regex ici...
REGEX = r"""
    (([0-99])+(\,\d{0,})?(\/[0-9])?
    (\snoix?)?(\senveloppe?)?(\tasse?)?(\smorceau?)?(\strait?)?)
    (\ ?(gousses|(B|b)ouquet|(R|r)ondelle|feuilles|tasse(s|)|
        cuillÃ¨re(s|)((\sÃ(\s*)cafÃ©)|
        (\sÃ(\s*)soupe))?|m(L|l)|(g\s)|l(b|B)|tranches|pintes|gallon|pincÃ©e|
        cl|oz|(c\.\s{0,}Ã\s{0,}((s\.)|(c\.)|(\.c)|(\.s)|thÃ©|soupe))))?|
    ((.*)\)|
    (U|u)ne pincÃ©e|(A|a)u goÃ»t|(Q|q)uelques|(F|f)euilles)
"""
REGEX_TEXT = r"^(de |d'|dâ€™|du|des)"
pattern = re.compile(REGEX, re.VERBOSE | re.IGNORECASE)
item_pattern = re.compile(REGEX_TEXT, re.VERBOSE | re.IGNORECASE)
#

def load_ingredients(filename):
    with open(filename, 'r', encoding='utf-8') as f:
        raw_items = f.readlines()
    ingredients = [x.strip() for x in raw_items]
    return ingredients


def get_ingredients(text):
    # Insérez ici votre code pour l'extraction d'ingrédients.
    # En entrée, on devrait recevoir une ligne de texte qui correspond à un ingrédient.
    # Par ex. 2 cuillères à café de poudre à pâte
    # Vous pouvez ajouter autant de fonctions que vous le souhaitez.
    #
    # IMPORTANT : Ne pas modifier la signature de cette fonction
    #             afin de faciliter notre travail de correction.
    #
    # Votre code ici...
    result = pattern.match(text)
    if (result != None):
        item = text[result.end():].strip()
        # 1_improvement - Before this little plus, we were at 20% succes rate
        result_item_pattern = item_pattern.match(item)
        if (result_item_pattern != None):
            item = item[result_item_pattern.end():].strip()
        return (result.group().strip(),fix_ingredient(item))
    else:
        return (fix_ingredient(text), "")
    #
    #return "2 cuillères à café", "poudre à pâte"   # À modifier - retourner la paire extraite
def fix_ingredient(item):
    return item.replace(
            ", pour dÃ©corer","").replace(
                "en purÃ©e", "").replace(
                    "rÃ¢pÃ©", "").replace(
                        " Ã©crasÃ©s", "").replace(
                            ", tranchÃ©","").replace(
                                "Ã©mincÃ©", "").replace(
                                    "au goÃ»t", "").replace(
                                        "(surtout pas en boÃ®te)", "").replace(
                                            " pelÃ©es, en tranches", "").replace(
                                                " battu", "").replace(
                                                    "tranchÃ©", "").replace(
                                                        "hachÃ©es", ""
                                    ).strip()


if __name__ == '__main__':
    # Vous pouvez modifier cette section
    print("Lecture des ingrédients du fichier {}. Voici quelques exemples: ".format(ingredients_fn))
    all_items = load_ingredients(ingredients_fn)
    for item in all_items[:5]:
        print("\t", item)
    print("\nExemples d'extraction")
    for item in all_items[:10]:
        quantity, ingredient = get_ingredients(item)
        print("\t{}\t QUANTITE: {}\t INGREDIENT: {}".format(item, quantity, ingredient))

    
    ingredients_text = get_file_content(get_complet_path("./ingredients.txt"))
    solutions_text = get_file_content(get_complet_path("./ingredients_solutions.txt"))
    print(len(ingredients_text))
    print(len(solutions_text))
    quantity_correct = 0
    quantity_wrong = 0
    solution_items=load_ingredients(solutions_fn)
    print("###################### SOLUTION")

    for count,ingredient_text in ingredients_text.items():
        ingredient = get_ingredients(ingredient_text)
        ingredient_solution = solutions_text[count].split("   ")
        print(count)
        solution_quantity = ingredient_solution[1].replace("QUANTITE:","")
        solution_item = ingredient_solution[2].replace("INGREDIENT:","")
        #print(solution_quantity)
        #print(ingredient[1])

        if (solution_quantity == ingredient[0] and solution_item == ingredient[1]):
            quantity_correct +=1
        else:
            quantity_wrong +=1
            print("NOT MATCH:\n {!r} \n Solution  (item='{}', quantity='{}')\n".format(ingredient, solution_item,solution_quantity))

    succes_rate = quantity_correct * 100 / len(ingredients_text)
    print("\nResult: CORRECT({}) x WRONG({}) - {:.2f}%".format(quantity_correct, quantity_wrong, succes_rate))
# In[ ]:





# In[ ]:




