def set_refresh_cookie(response,refresh:str):
    response.set_cookie(
        "refresh_token",
        refresh,
        httponly =True,
        secure =False,
        samesite="Lax",
        max_age=7*24*3600,
        path="/api/v1/auth",
    )

def clear_refresh_cookie(response):
    response.delete_cookie("refresh_token",path="/api/v1/auth")
    