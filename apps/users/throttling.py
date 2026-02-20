from rest_framework.throttling import UserRateThrottle, ScopedRateThrottle

class BurstRateThrottle(UserRateThrottle):
    scope = "burst"
    
class SustainedRateThrottle(UserRateThrottle):
    scope = "sustained"


class LoginBurstThrottle(ScopedRateThrottle):
    scope = "login_burst"
    
class LoginSustainedThrottle(ScopedRateThrottle):
    scope = "login_sustained"