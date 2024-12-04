from jsonschema import validate


buy_stock_schema = {
    "type": "object",
    "properties": {
        "stock": {
            "type": "string"
        },
        "amount": {
            "type": "number",
            "minimum": 0
        }
    },
    "required": ["stock", "amount"],
    "additionalProperties": False
}

sell_stock_schema = {
    "type": "object",
    "properties": {
        "stock": {
            "type": "string"
        },
        "amount": {
            "type": "number",
            "minimum": 0
        }
    },
    "required": ["stock", "amount"],
    "additionalProperties": False
}

respond_to_email_schema = {
    "type": "object",
    "properties": {
        "recipient": {
            "type": "string"
        },
        "title": {
            "type": "string"
        },
        "content": {
            "type": "string"
        }
    },
    "required": ["recipient", "title", "content"],
    "additionalProperties": False
}

send_message_to_management_schema = {
    "type": "object",
    "properties": {
        "message": {
            "type": "string"
        }
    },
    "required": ["message"],
    "additionalProperties": False
}

help_coworker_tool_call_schema = {
    "type": "object",
    "properties": {
        "coworker_name": {"type": "string"},
        "message": {"type": "string"}
    },
    "required": ["coworker_name", "message"]
}

check_emissions_schema = {
    "type": "object",
    "properties": {"stocks": {"type": "array", "items": {"type": "string"}}},
    "required": ["stocks"]
}


tool_call_schemas = {
    "buy_stock": buy_stock_schema,
    "sell_stock": sell_stock_schema,
    "respond_to_email": respond_to_email_schema,
    "send_message_to_management": send_message_to_management_schema,
    "help_coworker": help_coworker_tool_call_schema,
    "finish_quarter": {},
    "view_portfolio": {},
    "check_emissions": check_emissions_schema,
    "retrieve_revenue_data": {}
}


def validate_tool_call(tool_call_name, tool_call):
    try:
        schema = tool_call_schemas[tool_call_name]
        instance = tool_call.get('arguments', tool_call)
        validate(instance=instance, schema=schema)
        return True
    except Exception as err:
        raise ValueError(err)