import spacy


TEST_REVIEW = """Chan-wook Park, you have to hand it to the guy. In my eyes, he's not only a brilliant director but a 
brilliant director who can turn his hand to any genre and often provides something refreshing yet still ultimately 
satisfying.<br /><br />Thirst is, essentially, a vampire tale but one that plays fast and loose with some of the 
"rules" of the subgenre. Kang-ho Song plays Father Sang-hyeon, a man who unselfishly gives himself over to a research 
program and then unselfishly kind of catches the disease they are trying to cure, dies and comes back. All thanks to 
the blood he was transfused with. Being the only one out of five hundred to survive, he becomes quite the celebrity 
to those who know him and all he wants is to get back to normal. Normal, however, now involves being able to leap 
great distances without injury, wanting to drink blood and getting severely hot under the collar when rays of sun get 
on his skin. It's not long before he's living with a rather dysfunctional family unit who knew him in his childhood 
and while he hides his new, strange lifestyle he finds himself drawn into a complex love triangle, becoming more 
acceptable of darker thoughts and sliding down a slippery slope that could lead him from man to beast to monster.<br 
/><br />Deftly blending a number of genres, Park's movie felt much fresher and more original to me than Let The Right 
One In (to use a recent example) and genuinely impressed me with it's approach to material that could easily have 
felt as well-worn and rehashed as any number of other vampire movies we've seen over the years. It's a mixture of 
horror, melodrama and comedy while also pondering ideas such as strength of faith, the power over life and death, 
the downside of immortality, etc, etc.<br /><br />Some people have complained that this genre-blending approach 
weakens the movie but I personally found that it was a lively, entertaining and always enjoyable movie helped by a 
great central performance from Song as the tortured priest and fantastic turns from a supporting cast with no weak 
links. Many characters get to move through a range of emotions and all do so with skill and believability, 
especially the young woman (played by OK-vin Kim) who becomes the object of the priest's love, lust and affection.<br 
/><br />Fans of Asian cinema (and Park in particular) and also fans of Poe's "The Tell-tale Heart" (watch and learn) 
should lap this up, it's yet another classy movie from a man who seems to take everything in his stride and always 
manages to put out nothing less than solid entertainment.<br /><br />See this if you like: Cronos, Near Dark, 
Dellamorte Dellamore AKA Cemetery Man. """


def test_model(input_data: str = TEST_REVIEW):
    loaded_model = spacy.load("model_artifacts")
    # Generate prediction
    parsed_text = loaded_model(input_data)
    # Determine prediction to return
    if parsed_text.cats["pos"] > parsed_text.cats["neg"]:
        prediction = "Positive"
        score = parsed_text.cats["pos"]
    else:
        prediction = "Negative"
        score = parsed_text.cats["neg"]
    print(
        f"Review text: {input_data}\nPredicted sentiment: {prediction}"
        f"\tScore: {score}"
    )


print("Testing model")
test_model()