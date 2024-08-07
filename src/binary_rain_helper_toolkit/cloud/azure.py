import json
import azure.functions as func

def return_http_response(
    message: str,
    status_code: int
) -> func.HttpResponse:
    """
    Format an HTTP response.
    
    ### Parameters:
    ----------
    message : str
        The message to be returned in the response.
    status_code : int
        The status code of the response.

    ### Returns:
    ----------
    response : func.HttpResponse
        The formatted HTTP response
    """
    if str(status_code).startswith("2"):
        status = "OK"
    else:
        status = "NOK"

    return func.HttpResponse(
        json.dumps({
            "response": message,
            "status": status
        }),
        status_code=status_code,
        mimetype="application/json",
    )