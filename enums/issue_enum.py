from enum import Enum

class IssueEnum(Enum):
    NO_ISSUE_FOUND = "No Issue found"
    ISSUE_FOUND = "Issue found"
    FEATURE_NOT_FOUND = "Feature not found"

class SceneEnum(Enum):
    UI_INTERACTION = "UI Interaction"
    STATE_VERIFICATION = "State Verification"
    CONDITIONAL = "Conditional"
    NAVIGATION = "Navigation"
    WAITING = "Waiting"
    DESCRIPTIVE = "Descriptive"
    INPUT = "Input"
    SCROLL = "Scroll"