from langchain import FewShotPromptTemplate, PromptTemplate


######################## NER ###########################

prompt_template = lambda plus_plus : {
    "@@##" : PromptTemplate(
        input_variables=['tag','precision', 'sentence', 'few_shots'],
        template = """### SYSTEM : The task is to extract all the named entites that are {tag} in the following sentence.
### USER : Your goal is to add '@@' at the begining and '##' at the end of all the enities that are {tag}. {precision}.
{few_shots}\n
### ASSISTANT : Ok now I understand I need to rewrite the sentence and add '@@' at the begining and '##' at the end of all the enities that are {tag}. Can you now provide me the sentence ? 
### INPUT : <start_input> {sentence} <end_input>
### OUTPUT : <start_output> """),

    "discussion" : PromptTemplate(
        input_variables=['sentence', 'few_shots', 'precisions'],
        template = """### SYSTEM : The task is to extract all the named entites in the following sentence.
### USER : Your goal is to extract all the enities that are either person, organization, location or miscallaneous and output the entities in a list of tuples. In each tuple put the named entity and the tag alongside it.
{precisions}{few_shots}
### ASSISTANT : Ok now I understand I need to only output a list with the entities that are in the sentence and the tag along it. Can you now provide me the sentence ? 
### INPUT : <start_input> {sentence} <end_input>
### OUTPUT : <start_output> ["""),

    "wrapper" : PromptTemplate(
        input_variables=['sentence', 'few_shots', 'precisions'],
        template = """### SYSTEM : The task is to extract all the named entites in the following sentence.
### USER : Your goal is to extract all the entities that have either tag person, organization, location or miscallaneous. 
In order to do this, you have to rewrite the sentence and wrap the named entity by <tag> and </tag> where tag is either "person", "organization", "location" or "miscellaneous".
{precisions}{few_shots}
### ASSISTANT : Ok now I understand I need to rewrite the sentence and wrap the named entity by  <tag> and </tag> where tag is either person, organization, location or miscellaneous.. Can you now provide me the sentence ? 
### INPUT : <start_input> {sentence} <end_input>
### OUTPUT : <start_output> """),

    "get-entities" : PromptTemplate(
        input_variables=['sentence', 'few_shots', 'precisions'],
        template = """### SYSTEM : The task is to extract all the named entites in the following sentence.
### USER : Your goal is to extract all the entities that are either person, organization, location or miscallaneous and output the entities in a list. Output all entities even if you are not completely sure it is an entity.
{precisions}{few_shots}
### ASSISTANT : Ok now I understand I need to only output a list with the entities. Can you now provide me the sentence ? 
### INPUT : <start_input> {sentence} <end_input>
### OUTPUT : <start_output> ["""),

    "tagger" : PromptTemplate(
        input_variables=['entities_sentence', 'few_shots', 'precisions'],
        template = """### SYSTEM : The task is to tag all the named entites that were extracted from a sentence.
### USER : Your task is to to tag all the named entites that were extracted from a sentence with either 
    'P' for person entities, 
    'O' for organization entities, 
    'L' for location entities,
    'M' for miscallaneous entities or
    'N' is it is none of the above.
Output a json with the named entities as keys and the tag as values.
{precisions}{few_shots}
### ASSISTANT : Ok now I understand I need to only output json with the named entities as keys and the tag as values. Can you now provide me the list of extracted entities and the sentence ? 
### INPUT : <start_input> {entities_sentence} <end_input>
### OUTPUT : <start_output> {{ """),

}

prompt_template_ontonotes = lambda plus_plus : {
    "@@##" : PromptTemplate(
        input_variables=['tag', 'sentence', 'few_shots', 'precision'],
        template = get_system_start("ontonote5", plus_plus) + """
### USER : I want to to extract all the named entities that are of type {tag}. {tag} are {precision}
### ASSISTANT : What is the format of the output ?
### USER : You have to rewrite the sentence that I gave you and wrap the named entities of type {tag} by writting '@@' at the begining and '##' at the end. For exemple, with the sentence "Japan is a country" as input and asking for tag GPE, you would answer "@@Japan## is a country". 
{few_shots}\n
### ASSISTANT : I will extract all the named entities that have tag {tag} by writting '@@' at the begining and '##' at the end of all these entities. Can you now provide me the sentence ? 
### INPUT : <start_input> {sentence} <end_input>
### OUTPUT : <start_output> """),

    "discussion" : PromptTemplate(
        input_variables=['sentence', 'few_shots', 'precisions'],
        template = get_system_start("ontonote5", plus_plus) + """
{precisions}### USER : I want you to extract all the named entities in the text and tag them with one of the tag of the OntoNote5 dataset.
### ASSISTANT : What is the format of the output ?
### USER : You have to output a python list of tuples. In each tuple, you have a the named entity and its own tag. For exemple, with the sentence "Japan is a country" as input, you would answer "[('Japan', 'GPE')]". 
{few_shots}
### ASSISTANT : I will extract all the named entities in the text that has one of the 18 tags of the OntoNote5 dataset and output a python list of tuples containing the named entitiy and its own tag. Now provide me the sentence.
### INPUT : <start_input> {sentence} <end_input>
### OUTPUT : <start_output> ["""),

    "wrapper" : PromptTemplate(
        input_variables=['sentence', 'few_shots', 'precisions'],
        template = get_system_start("ontonote5", plus_plus) + """
{precisions}### USER : I want you to extract all the named entities in the text and tag them with one of the tag of the OntoNote5 dataset.
### ASSISTANT : What is the format of the output ?
### USER : As output, you have to rewrite the sentence that I gave you and wrap all the named entities with "<[tag]>" and "</[tag]>" where "[tag]" is one of the 18 tags of the OntoNote5 dataset. For exemple, with the sentence "Japan is a country" as input, you would answer "<GPE>Japan</GPE> is a country". 
{few_shots}
### ASSISTANT : I understand I need to rewrite the sentence and wrap the named entity with"<[tag]>" and "</[tag]>" where "[tag]" is one of the 18 tags of the OntoNote5 dataset. Now, provide me the sentence ? 
### INPUT : <start_input> {sentence} <end_input>
### OUTPUT : <start_output> """),

    "get-entities" : PromptTemplate(
        input_variables=['sentence', 'few_shots', 'precisions'],
        template = get_system_start("ontonote5", plus_plus) + """
{precisions}### USER : I want you to extract all the named entities in the text and tag them with one of the tag of the OntoNote5 dataset.
### ASSISTANT : What is the format of the output ?
### USER : I want you to output a python list that contains all the named entities in the sentence. For exemple, with the sentence "Japan is a country" as input, you would answer "["Japan"]". 
{few_shots}
### ASSISTANT : Ok now I understand I need to only output a list with all the named entities. Now provide me the sentence. 
### INPUT : <start_input> {sentence} <end_input>
### OUTPUT : <start_output> ["""),

    "tagger" : PromptTemplate(
        input_variables=['entities_sentence', 'few_shots', 'precisions'],
        template = get_system_start("ontonote5", plus_plus, start = "### SYSTEM : The task is to tag all the named entites that were extracted from a sentence.")+"""
### USER : I want you to tag all the named entites that were extracted from a sentence with the following character :  
    '1' for 'CARDINAL' entities,{precisions}
    '2' for 'ORDINAL' entities,
    '3' for 'WORK_OF_ART' entities,
    '4' for 'PERSON' entities,
    '5' for 'LOC' entities,
    '6' for 'DATE' entities,
    '7' for 'PERCENT' entities,
    '8' for 'PRODUCT' entities,
    '9' for 'MONEY' entities,
    '0' for 'FAC' entities,
    'A' for 'TIME' entities,
    'B' for 'ORG' entities,
    'C' for 'QUANTITY' entities,
    'D' for 'LANGUAGE' entities,
    'E' for 'GPE' entities,
    'F' for 'LAW' entities,
    'G' for 'NORP' entities,
    'H' for 'EVENT' entities
### ASSISTANT : What is the format of the output ?
### USER : Output a json with the named entities as keys and the tag as values. For exemple, with the input '"[Japan]" in "Japan is a country", you would answer '{{ "Japan" : "E" }}'. 
{few_shots}
### ASSISTANT : I will extract all the entities in the sentence you will give me and add one of the 18 charachter corresponding to one of the tags. Now provide me the list of extracted entities and the sentence ? 
### INPUT : <start_input> "{entities_sentence}" <end_input>
### OUTPUT : <start_output> {{ """),

}

shot_prompt = PromptTemplate(
    input_variables=["text", "output_text"], 
    template="### INPUT : <start_input> {text} <end_input>\n### OUTPUT : <start_output> {output_text} <end_output>")

few_shot_prompt = lambda examples : FewShotPromptTemplate(
        examples=examples, 
        example_prompt=shot_prompt,
        suffix="", 
        input_variables=[])

###################### Verifiers ########################

verifier_prompt_template = PromptTemplate.from_template(
    """### SYSTEM : The task is to verify whether the word is a {tag} extracted from the given sentence. {precision} 
    ### ASSISTANT : I will be happy to help you. Can you provide me some examples.
    ### USER : Here are some examples :\n{few_shots}""")

verifier_shot_prompt = PromptTemplate(input_variables=["sentence", "named_entity", "tag", "answer"], 
                                      template="""### INPUT SENTENCE : <start_sentence> {sentence} <end_sentence>
### USER : Is the word(s) "{named_entity}" in the input sentence a {tag} ? Please answer with "yes" or "no".
### ASSISTANT : <start_answer> {answer} <end_answer>""")

verifier_few_shot_prompt = lambda examples : FewShotPromptTemplate(
        examples=examples, 
        example_prompt=verifier_shot_prompt, 
        suffix="""### INPUT SENTENCE : <start_sentence> {sentence} <end_sentence>
### USER : Is the word(s) "{named_entity}" in the input sentence a {tag} ? Please answer with "yes" or "no".
### ASSISTANT : <start_answer> """, 
        input_variables=["sentence", 'tag', 'named_entity']
    )    

################## Confidence Checker ####################

confidence_prompt_template = PromptTemplate(
        input_variables=['entities_sentence', 'precisions'],
        template = """### SYSTEM : The task is to give a confidence on the tag assigned to a named entity.
### USER : You will recevie a sentence and a python list of the named entities that were extracted from this sentence with their tag. You task is to assign one of the following 5 confidence level on the fact that the named entity is in fact a named entity of this particular tag.
The five confidence levels are "low", "medium-low", "medium", "medium-high", "high". 
### ASSISTANT : How would you like me to output the confidences ? 
### USER : Return a json with the name of the named entity as key and the confidence level as value
{precisions}
### ASSISTANT : Ok now I understand I need to only output json with the named entities as keys and as values one of the 5 confidence levels that represents for each entry (named entity, tag) the confidence I have on the fact that I assigned this specific tag to this entity. Can you now provide me the sentence and the list of extracted entities? 
### INPUT : <start_input> {entities_sentence} <end_input>
### OUTPUT : <start_output> {{ """)

###################### Mappings ########################

precision_ner = {'PER' : 'Person entities are all the names you can find in the text. That can be celebrities, historical figures, fictional characters or just names but not pronouns like "he" or "she".', 
                 'ORG' : 'Organization entities all the organizations you can find in the text. That can be business, educational organisation, broadcaster, sports organisation, scientific organisation, political organisation, institute or government agency.',
                 'LOC' : 'Location entities are all countries, the human-geographic territorials, geographical regions, areas in a single country or natural geographic objects.', 
                 'MISC': 'Miscellaneous entities are all events, languages, adjectives to describe things particular to a country. It cannot be verbs, numbers or time related word like weekdays and months. '}
mapping_abbr_string_ner = {'PER' : 'person entities', 'ORG' : 'organization entities', 'LOC' : 'location entities', 'MISC': 'miscellaneous entities (i.e. entities that are not person, organization or location)',
                           'CARDINAL' : 'CARDINAL entities',
                           'ORDINAL' : 'ORDINAL entities',
                           'WORK_OF_ART' : 'WORK_OF_ART entities',
                           'PERSON' : 'PERSON entities',
                           'DATE' : 'DATE entities',
                           'PERCENT' : 'PERCENT entities',
                           'PRODUCT' : 'PRODUCT entities',
                           'MONEY' : 'MONEY entities',
                           'FAC' : 'FAC entities',
                           'TIME' : 'TIME entities',
                           'QUANTITY' : 'QUANTITY entities',
                           'LANGUAGE' : 'LANGUAGE entities',
                           'GPE' : 'GPE entities',
                           'LAW' : 'LAW entities',
                           'NORP' : 'NORP entities',
                           'EVENT' : 'EVENT entities'}
mapping_abbr_string_verifier = {'PER' : 'person entity', 'ORG' : 'organization entity', 'LOC' : 'location entity', 'MISC': 'miscellaneous entity (i.e. en entity that is not a tperson, an organization or a location)'}
mapping_string_abbr = {'person' : 'PER', 'organization' : 'ORG', 'location' : 'LOC', 'miscellaneous': "MISC"}
mapping_tag_words = {'PER' : 'person', 'ORG' : 'organisation', 'LOC' : 'location', 'MISC' : 'miscellaneous'}

###################### SYSTEM START #####################

def get_system_start(dataset : str, plus_plus = False, start = "### SYSTEM : The task is to extract named entites in a sentence." ):
    if dataset == 'ontonote5' :
        if plus_plus :
            return start + """
    A named entity refers to a specific, named object, concept, location, person, organization, or other entities that have a proper name. Named entities are typically unique and distinguishable entities that can be explicitly named or referred to in text. 
    The goal of named entity extraction is to identify and classify these entities within a given text.
    We are working with 18 types of entities of the OntoNote5 dataset that are listed below with their description :
        "CARDINAL": "Numerals that do not fall under another type.",
        "ORDINAL": "Words or expressions indicating order.",
        "WORK_OF_ART": "Titles of creative works.",
        "PERSON": "Names of people, including fictional and real characters.",
        "LOC": "Geographical locations, both physical and political.",
        "DATE": "Temporal expressions indicating dates or periods.",
        "PERCENT": "Percentage values.",
        "PRODUCT": "Names of products or services.",
        "MONEY": "Monetary values, including currency symbols.",
        "FAC": "Named facilities, such as buildings, airports, or highways.",
        "TIME": "Temporal expressions indicating times of the day.",
        "ORG": "Names of organizations, institutions, or companies.",
        "QUANTITY": "Measurements or counts, including units.",
        "LANGUAGE": "Names of languages.",
        "GPE": "Geopolitical entities, such as countries, cities, or states.",
        "LAW": "Legal references, including laws and legal concepts.",
        "NORP": "Nationalities, religious, or political groups.",
        "EVENT": "Named occurrences or incidents."""
        else : 
            return start + """
    The types of the entities have to be one of the OntoNote5 dataset that you can find here : ['CARDINAL', 'ORDINAL', 'WORK_OF_ART', 'PERSON', 'LOC', 'DATE', 'PERCENT', 'PRODUCT', 'MONEY', 'FAC', 'TIME', 'ORG', 'QUANTITY', 'LANGUAGE', 'GPE', 'LAW', 'NORP', 'EVENT']."""
    else : 
        return  ""
    
