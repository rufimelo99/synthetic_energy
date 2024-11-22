from enum import Enum


class AnonymizationType(Enum):
    NAME = "name"
    EMAIL = "email"
    PHONE_NUMBER = "phone_number"
    ADDRESS = "address"
    CREDIT_CARD_NUMBER = "credit_card_number"
    CREDIT_CARD_PROVIDER = "credit_card_provider"
    CREDIT_CARD_SECURITY_CODE = "credit_card_security_code"
    CREDIT_CARD_EXPIRATION_DATE = "credit_card_expiration_date"
    CREDIT_CARD_FULL = "credit_card_full"
    COMPANY = "company"
    SSN = "ssn"
    IPV4 = "ipv4"
    IPV6 = "ipv6"
    URL = "url"
    NON_SENSIBLE_DATA = "non_sensible_data"
    OTHER = "other"  # For custom anonymization types that are not being handled by the library.
