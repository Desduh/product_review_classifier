import pandas as pd

# Dados de treinamento (15 revisões, 5 de cada classe)
# train_data = {
#     "review": [
#         "Great product, works as expected",  # Positive
#         "I love this, highly recommended",  # Positive
#         "Excellent quality and fast delivery",  # Positive
#         "Amazing experience, will buy again",  # Positive
#         "Very satisfied with my purchase",  # Positive
#         "Not good, broke after a week",  # Negative
#         "Terrible service, would not recommend",  # Negative
#         "Very poor quality, disappointed",  # Negative
#         "Product did not match the description",  # Negative
#         "A waste of money",  # Negative
#         "It's okay, not great but not bad either",  # Neutral
#         "Average quality, nothing special",  # Neutral
#         "Decent product, met my expectations",  # Neutral
#         "Satisfactory, but could be better",  # Neutral
#         "Neither good nor bad, just fine",  # Neutral
#     ],
#     "label": [
#         "positive", "positive", "positive", "positive", "positive",
#         "negative", "negative", "negative", "negative", "negative",
#         "neutral", "neutral", "neutral", "neutral", "neutral",
#     ]
# }

train_data = {
    "review": [
        "Great product, works as expected",  # Positive
        "I love this, highly recommended",  # Positive
        "Excellent quality and fast delivery",  # Positive
        "Amazing experience, will buy again",  # Positive
        "Very satisfied with my purchase",  # Positive
        "It's okay, not great but not bad either",  # Neutral
        "Average quality, nothing special",  # Neutral
        "Decent product, met my expectations",  # Neutral
        "Satisfactory, but could be better",  # Neutral
        "Neither good nor bad, just fine",  # Neutral
        "Great value for the price",  # Neutral
        "Good, but has some flaws",  # Neutral
        "Not impressed, expected better",  # Neutral
        "Terrible service, would not recommend",  # Negative
        "Not good, broke after a week",  # Negative
    ],
    "label": [
        "positive", "positive", "positive", "positive", "positive",
        "neutral", "neutral", "neutral", "neutral", "neutral", 
        "neutral", "neutral", "neutral", "negative", "negative"
    ]
}

# Dados de teste (45 revisões, 15 de cada classe)
test_data = {
    "review": [
        # Positivos
        "I absolutely love this product",
        "Exceeded my expectations, fantastic quality",
        "Very happy with my purchase",
        "Amazing value for the price",
        "Highly recommend, great experience",
        "Excellent product, will buy again",
        "Superb, can't complain",
        "Great quality, worth every penny",
        "Best product I've bought this year",
        "Perfect for my needs, outstanding",
        "Wonderful and reliable",
        "Outstanding performance, highly satisfied",
        "Love it, amazing features",
        "Top-notch quality, worth the price",
        "Exceptional service and product",
        
        # Negativos
        "Horrible experience, do not buy",
        "Very disappointed, bad quality",
        "Arrived broken, waste of money",
        "Worst purchase I've ever made",
        "Absolutely terrible, don't recommend",
        "Does not work as advertised",
        "Poor quality, very disappointed",
        "Failed after one use, awful",
        "Totally unsatisfied with the product",
        "Cheap and unreliable, regret buying",
        "Useless, does not meet expectations",
        "Awful service and product",
        "Broken and unusable",
        "Worst quality ever",
        "Unacceptable, waste of money",
        
        # Neutros
        "It's okay, nothing special",
        "Neither good nor bad, just average",
        "Satisfactory but could be improved",
        "Decent product, met some expectations",
        "Not impressed but not terrible either",
        "Mediocre, could be better",
        "Just fine, nothing extraordinary",
        "Average quality, works fine",
        "Neutral experience, as expected",
        "Met expectations but not great",
        "Okay product for the price",
        "Fair quality, no complaints",
        "Standard product, does the job",
        "Nothing to write home about",
        "Acceptable, but not amazing",
    ],
    "label": [
        "positive"] * 15 +
        ["negative"] * 15 +
        ["neutral"] * 15
}

train_df = pd.DataFrame(train_data)
test_df = pd.DataFrame(test_data)

train_df.to_csv("data/reviews_train.csv", index=False)
test_df.to_csv("data/reviews_test.csv", index=False)