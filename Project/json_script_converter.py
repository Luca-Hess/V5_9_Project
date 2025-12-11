from typing import List, Dict


# Define automation machine capabilities
class LabAutomationMachine:
    """Defines the capabilities of a fictional lab automation machine"""

    AUTOMATED_ACTIONS = {
        'pipette': {'params': ['source', 'destination', 'volume']},
        'add': {'params': ['reagent', 'container', 'volume']},
        'mix': {'params': ['container', 'duration', 'speed']},
        'incubate': {'params': ['container', 'temperature', 'duration']},
        'centrifuge': {'params': ['container', 'speed', 'duration', 'temperature']},
        'heat': {'params': ['container', 'temperature', 'duration']},
        'cool': {'params': ['container', 'temperature', 'duration']},
        'dispense': {'params': ['reagent', 'destination', 'volume']},
        'aspirate': {'params': ['source', 'volume']},
        'shake': {'params': ['container', 'duration', 'speed']},
        'wait': {'params': ['duration']},
    }

    @classmethod
    def get_action_type(cls, action: str) -> str:
        """Determine if action is automated or manual"""
        action_lower = action.lower()
        if action_lower in cls.AUTOMATED_ACTIONS:
            return 'automated'
        return 'manual' # Default to manual if unknown actions


class JsonScriptConverter:
    """Converts annotation format to automation script format"""

    def __init__(self):
        self.machine = LabAutomationMachine()

    def generate_script(self, actions: List[Dict]) -> str:
        """Generate automation script from actions"""
        lines = [
            "# Lab Automation Script",
            "# Generated from protocol analysis",
            ""
        ]
        current_mode = None

        for action in actions:
            automatable = bool(action.get("automatable", False))
            name = (action.get('action') or 'unknown').lower()
            params = action.get('params') or action.get('parameters') or {}

            # Add section headers when switching between manual/automated
            if automatable != current_mode:
                lines.append("\n### AUTOMATED SECTION ###" if automatable else
                             "\n### MANUAL SECTION ###")
                current_mode = automatable

            if automatable and name in self.machine.AUTOMATED_ACTIONS:
                clean = {
                    k: v for k, v in params.items()
                    if v not in (None, {}, []) and k != "other"
                }
                other = params.get("other") or {}
                clean.update({k: v for k, v in other.items() if v is not None})

                param_str = ", ".join([f"{k}={repr(v)}" for k, v in clean.items()])
                lines.append(f"{name}({param_str})")
            else:
                desc = ", ".join(
                    f"{k}: {v}" for k, v in params.items() if v is not None
                )
                lines.append(f"# MANUAL: {name} ({desc})")
        return "\n".join(lines)