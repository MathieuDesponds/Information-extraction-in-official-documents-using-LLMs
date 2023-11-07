from langchain import FewShotPromptTemplate, PromptTemplate


######################## NER ###########################

prompt_template = {
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
### OUTPUT : <start_output> """),
    "<>" : PromptTemplate(
        input_variables=['sentence', 'few_shots', 'precisions'],
        template = """### SYSTEM : The task is to extract all the named entites in the following sentence.
### USER : Your goal is to extract all the entities that have either tag person, organization, location or miscallaneous. 
In order to do this, you have to rewrite the sentence and wrap the named entity by <tag> and </tag>.
{precisions}{few_shots}
### ASSISTANT : Ok now I understand I need to rewrite the sentence and wrap the named entity by <tag> and </tag>. Can you now provide me the sentence ? 
### INPUT : <start_input> {sentence} <end_input>
### OUTPUT : <start_output> """)
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

###################### Mappings ########################

precision_ner = {'PER' : 'Person entities are all the names you can find in the text. That can be celebrities, historical figures, fictional characters or just random names.', 
                 'ORG' : 'Organization entities all the organizations you can find in the text. That can be business, educational organisation, broadcaster, sports organisation, scientific organisation, political organisation, institute or government agency.',
                 'LOC' : 'Location entities are all the human-geographic territorials, geographical regions, areas in a single country or natural geographic objects.', 
                 'MISC': 'Miscellaneous entities are all events or, names, entities and adjectives that are specific but do not have a well-defined category and do not fit in person, organization or location entities'}
mapping_abbr_string_ner = {'PER' : 'person entities', 'ORG' : 'organization entities', 'LOC' : 'location entities', 'MISC': 'miscellaneous entities (i.e. entities that are not person, organization or location)'}
mapping_abbr_string_verifier = {'PER' : 'person entity', 'ORG' : 'organization entity', 'LOC' : 'location entity', 'MISC': 'miscellaneous entity (i.e. en entity that is not a tperson, an organization or a location)'}
mapping_string_abbr = {'person' : 'PER', 'organization' : 'ORG', 'location' : 'LOC', 'miscellaneous': "MISC"}
