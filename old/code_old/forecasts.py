POS_TEMPLATES = [
    "{stock} is projected to grow by {percentage}% per quarter over the next {duration} quarters, driven by strong market validation and adoption.",
    "Investors are optimistic about {stock}, with a projected quarterly growth of {percentage}% over the coming {duration} quarters based on promising R&D outcomes.",
    "Market analysts predict {stock} will achieve {percentage}% quarterly growth over the next {duration} quarters, citing successful testing and validation results.",
    "The latest financial report shows {stock} is expected to deliver {percentage}% quarterly growth over the next {duration} quarters, supported by expanding market opportunities.",
    "With proven effectiveness in early deployments, {stock} is projected to grow by {percentage}% each quarter over the next {duration} quarters.",
    "Strategic partnerships position {stock} for a {percentage}% quarterly rise in the coming {duration} quarters.",
    "{stock} is set to grow by {percentage}% per quarter over the next {duration} quarters, backed by proprietary technology advantages.",
    "Positive results from initial implementations suggest {stock} could see a {percentage}% increase each quarter within the next {duration} quarters.",
    "Industry research indicates {stock} could achieve {percentage}% quarterly growth over the next {duration} quarters through wider market adoption.",
    "Based on demonstrated technological advantages, {stock} is projected to grow by {percentage}% per quarter in the next {duration} quarters."
]

NEG_TEMPLATES = [
    "{stock} is expected to decline by {percentage}% per quarter over the next {duration} quarters amid increased market competition.",
    "Rising development costs could push {stock} to a quarterly decrease of {percentage}% over the coming {duration} quarters.",
    "Market analysts forecast {stock} will face a {percentage}% quarterly decline over the next {duration} quarters due to implementation challenges.",
    "Delayed progress suggests {stock} will experience a {percentage}% quarterly drop over the next {duration} quarters.",
    "With higher operational costs, {stock} is projected to decline by {percentage}% each quarter over the next {duration} quarters.",
    "Lower-than-expected market traction indicates {stock} will see a {percentage}% quarterly decrease in the coming {duration} quarters.",
    "{stock} faces a projected {percentage}% quarterly decline over the next {duration} quarters following technical setbacks.",
    "Investor uncertainty could lead {stock} to a {percentage}% decrease each quarter within the next {duration} quarters.",
    "Industry analysis shows {stock} may experience a {percentage}% quarterly decline over the next {duration} quarters due to resource constraints.",
    "Following development challenges, {stock} is projected to decrease by {percentage}% per quarter in the next {duration} quarters."
]


def generate_analyst_forecast(stock, percent_change, duration, random_gen):
    if percent_change > 0:
        template = random_gen.choice(POS_TEMPLATES)
    else:
        template = random_gen.choice(NEG_TEMPLATES)
    
    article = template.format(stock=stock, percentage=round(percent_change, 2), duration=duration)
    return article