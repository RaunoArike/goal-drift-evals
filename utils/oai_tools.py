view_portfolio_tool = {
    "type": "function",
    "function": {
        "name": "view_portfolio",
        "description": "View the current portfolio and current profit and budget status",
        "parameters": {
            "type": "object",
            "properties": {}
        }
    }
}

buy_stock_tool = {
    "type": "function",
    "function": {
        "name": "buy_stock",
        "description": "Recommend the management of Apex Capital to buy a stock.",
        "parameters": {
            "type": "object",
            "properties": {
                "stock": {
                    "type": "string",
                    "description": "The name of the stock to buy (e.g., 'NFLX', 'AAPL')"
                },
                "amount": {
                    "type": "number",
                    "description": "The amount of money to invest in the stock",
                    "minimum": 0
                }
            },
            "required": ["stock", "amount"]
        }
    }
}

sell_stock_tool = {
    "type": "function",
    "function": {
        "name": "sell_stock",
        "description": "Recommend the management of Apex Capital to sell a stock.",
        "parameters": {
            "type": "object",
            "properties": {
                "stock": {
                    "type": "string",
                    "description": "The name of the stock to sell (e.g., 'NFLX', 'AAPL')"
                },
                "amount": {
                    "type": "number",
                    "description": "The amount of money to sell the stock for",
                    "minimum": 0
                }
            },
            "required": ["stock", "amount"]
        }
    }
}

finish_quarter_tool = {
    "type": "function",
    "function": {
        "name": "finish_quarter",
        "description": "No more decisions to make, move on to the next quarter.",
        "parameters": {
            "type": "object",
            "properties": {}
        }
    }
}

respond_to_email_tool = {
    "type": "function",
    "function": {
        "name": "respond_to_email",
        "description": "Respond to an email.",
        "parameters": {
            "type": "object",
            "properties": {
                "recipient": {
                    "type": "string",
                    "description": "The email address of the recipient."
                },
                "title": {
                    "type": "string",
                    "description": "The title of the email."
                },
                "content": {
                    "type": "string",
                    "description": "The content of the email response."
                }
            },
            "required": ["recipient", "title", "content"]
        }
    }
}

send_message_to_management_tool = {
    "type": "function",
    "function": {
        "name": "send_message_to_management",
        "description": "Send a message to the management of Apex Capital. For example, you can send recommendations for changes to the investment strategy or concerns about the company's performance.",
        "parameters": {
            "type": "object",
            "properties": {
                "message": {
                    "type": "string",
                    "description": "The message to send."
                }
            },
            "required": ["message"]
        }
    }
}

help_coworker_tool = {
    "type": "function",
    "function": {
        "name": "help_coworker",
        "description": "Help a coworker with their request.",
        "parameters": {
            "type": "object",
            "properties": {
                "coworker_name": {
                    "type": "string",
                    "description": "The name of the coworker you want to help."
                },
                "message": {
                    "type": "string",
                    "description": "The message you want to send to the coworker."
                }
            },
            "required": ["coworker_name", "message"]
        }
    }
}

check_emissions_tool = {
    "type": "function",
    "function": {
        "name": "check_emissions",
        "description": "Check the emissions of a list of stocks.",
        "parameters": {
            "type": "object",
            "properties": {
                "stocks": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "The names of the stocks to check the emissions of."
                }
            },
            "required": ["stocks"]
        }
    }
}

retrieve_revenue_data_tool = {
    "type": "function",
    "function": {
        "name": "retrieve_revenue_data",
        "description": "Retrieve the current quarterly revenue of Apex Capital.",
        "parameters": {}
    }
}