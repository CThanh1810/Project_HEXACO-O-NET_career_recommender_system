from sentence_transformers import SentenceTransformer, util
import torch

hexaco_facets = {
    "Sincerity":"a tendency to be genuine in interpersonal relations. Low scorers will flatter others or pretend to like them in order to obtain favors, whereas high scorers are unwilling to manipulate others",
    "Fairness":"a tendency to avoid fraud and corruption. Low scorers are willing to gain by cheating or stealing, whereas high scorers are unwilling to take advantage of other individuals or of society at large",
    'Greed Avoidance':'a tendency to be uninterested in possessing lavish wealth, luxury goods, and signs of high social status. Low scorers want to enjoy and to display wealth and privilege, whereas high scorers are not especially motivated by monetary or social-status considerations',
    'Modesty':'a tendency to be modest and unassuming. Low scorers consider themselves as superior and as entitled to privileges that others do not have, whereas high scorers view themselves as ordinary people without any claim to special treatment',
    'Fearfulness':'a tendency to experience fear. Low scorers feel little fear of injury and are relatively tough, brave, and insensitive to physical pain, whereas high scorers are strongly inclined to avoid physical harm',
    'Anxiety':'a tendency to worry in a variety of contexts. Low scorers feel little stress in response to difficulties, whereas high scorers tend to become preoccupied even by relatively minor problems',
    'Dependence':"one's need for emotional support from others. Low scorers feel self-assured and able to deal with problems without any help or advice, whereas high scorers want to share their difficulties with those who will provide encouragement and comfort",
    "Sentimentality":"a tendency to feel strong emotional bonds with others. Low scorers feel little emotion when saying good-bye or in reaction to the concerns of others, whereas high scorers feel strong emotional attachments and an empathic sensitivity to the feelings of others",
    "Social Self-Esteem":"a tendency to have positive self-regard, particularly in social contexts. Low scorers tend to have a sense of personal worthlessness and to see themselves as unpopular, whereas high scorers are generally satisfied with themselves and consider themselves to have likable qualities",
    "Social Boldness":"one's comfort or confidence within a variety of social situations. Low scorers feel shy or awkward in positions of leadership or when speaking in public, whereas high scorers are willing to approach strangers and are willing to speak up within group settings",
    "Sociability":"a tendency to enjoy conversation, social interaction, and parties. Low scorers generally prefer solitary activities and do not seek out conversation, whereas high scorers enjoy talking, visiting, and celebrating with others",
    "Liveliness":"one's typical enthusiasm and energy. Low scorers tend not to feel especially cheerful or dynamic, whereas high scorers usually experience a sense of optimism and high spirits",
    "Forgivingness":"one's willingness to feel trust and liking toward those who may have caused one harm. Low scorers tend 'hold a grudge' against those who have offended them, whereas high scorers are usually ready to trust others again and to re-establish friendly relations after having been treated badly",
    "Gentleness":"a tendency to be mild and lenient in dealings with other people. Low scorers tend to be critical in their evaluations of others, whereas high scorers are reluctant to judge others harshly",
    "Flexibility":"one's willingness to compromise and cooperate with others. Low scorers are seen as stubborn and are willing to argue, whereas high scorers avoid arguments and accommodate others' suggestions, even when these may be unreasonable",
    "Patience":"a tendency to remain calm rather than to become angry. Low scorers tend to lose their tempers quickly, whereas high scorers have a high threshold for feeling or expressing anger",
    "Organization":"a tendency to seek order, particularly in one's physical surroundings. Low scorers tend to be sloppy and haphazard, whereas high scorers keep things tidy and prefer a structured approach to tasks",
    "Diligence":"a tendency to work hard. Low scorers have little self-discipline and are not strongly motivated to achieve, whereas high scorers have a strong 'work ethic' and are willing to exert themselves",
    "Perfectionism":"a tendency to be thorough and concerned with details. Low scorers tolerate some errors in their work and tend to neglect details, whereas high scorers check carefully for mistakes and potential improvements",
    "Prudence":"a tendency to deliberate carefully and to inhibit impulses. Low scorers act on impulse and tend not to consider consequences, whereas high scorers consider their options carefully and tend to be cautious and self-controlled",
    "Aesthetic Appreciation":"one's enjoyment of beauty in art and in nature. Low scorers tend not to become absorbed in works of art or in natural wonders, whereas high scorers have a strong appreciation of various art forms and of natural wonders",
    "Inquisitiveness":"a tendency to seek information about, and experience with, the natural and human world. Low scorers have little curiosity about the natural or social sciences, whereas high scorers read widely and are interested in travel",
    "Creativity":"one's preference for innovation and experiment. Low scorers have little inclination for original thought, whereas high scorers actively seek new solutions to problems and express themselves in art",
    "Unconventionality":"a tendency to accept the unusual. Low scorers avoid eccentric or nonconforming persons, whereas high scorers are receptive to ideas that might seem strange or radical"
}

onet_workstyles = {
    "Achievement Orientation":"A tendency to establish and maintain personally challenging work-related goals, set high work related standards, and exert high effort toward meeting those goals and standards",
    "Attention to Detail":"A tendency to be detailed oriented, organized, and thorough in completing work",
    "Cautiousness":"A tendency to be careful, deliberate, and risk avoidant when making work-related decisions or doing work",
    "Dependability":"A tendency to be reliable, responsible, and consistently meet work-related obligations",
    "Self-Confidence":"A tendency to believe in one's work-related capabilities and ability to control one's work related outcomes",
    "Leadership Orientation":"A tendency to lead, take charge, offer opinions, and provide direction at work",
    "Social Orientation":"A tendency to seek out, enjoy, and be energized by social interaction at work",
    "Cooperation":"A tendency to be pleasant, helpful, and willing to assist others at work",
    "Empathy":"A tendency to show concern for others and be sensitive to others' needs and feelings at work",
    "Humility":"A tendency to be modest and humble when interacting with others at work",
    "Integrity":"A tendency to be honest and ethical at work",
    "Sincerity":"A tendency to be genuine and sincere in interactions with others at work, without concern for personal gain or self-interest",
    "Self-Control":"A tendency to remain calm, composed, and manage emotions effectively in response to criticism or difficult situations at work",
    "Stress Tolerance":"A tendency to cope and function effectively in stressful situations at work",
    "Adaptability":"A tendency to be open to and comfortable with change, new experiences, or ideas at work",
    "Innovation":"A tendency to be inventive, imaginative, and adopt new perspectives on ways to accomplish work",
    "Intellectual Curiosity":"A tendency to seek out and acquire new work-related knowledge and obtain a deep understanding of work-related subjects",
    "Tolerance for Ambiguity":"A tendency to be comfortable with ambiguity and uncertainty at work",
    "Initiative":"A tendency to be proactive and take on extra responsibilities and tasks that may fall outside of one's required work role",
    "Optimism":"A tendency to exhibit a positive attitude and positive emotions at work, even under difficult circumstances",
    "Perseverance":"A tendency to exhibit determination and resolve to perform or complete tasks in the face of difficult circumstances or obstacles at work"
}

hexaco_texts = list(hexaco_facets.values())
onet_texts = list(onet_workstyles.values())

model = SentenceTransformer('all-mpnet-base-v2')
hexaco_embeddings = model.encode(hexaco_texts, convert_to_tensor = True)
onet_embeddings = model.encode(onet_texts, convert_to_tensor=True)

similarity_matrix = util.cos_sim(hexaco_embeddings, onet_embeddings)
weight_matrix = torch.clamp(similarity_matrix, min=0)

col_sums = torch.sum(weight_matrix, dim=0)
normalized_weights = weight_matrix / col_sums

def predict_workstyles(user_scores_list):
    user_tensor = torch.tensor(user_scores_list)
    predicted_scores = torch.matmul(user_tensor, normalized_weights)
    return dict(zip(onet_workstyles.keys(), predicted_scores.tolist()))

example_user_scores = [4.1,3.28,4.25,3.94,6.58,4.84,3.28,5.04,3.8,3.52,3.94,4.65,4.68,4.01,5.28,4.24,3.91,4.25,4.36,4.08,3.62,4.57,4.14,3.59]
result = predict_workstyles(example_user_scores)
print("\nKết quả:")
for ws, score in result.items():
    print(f"{ws}: {score:.2f}")