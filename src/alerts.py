from src.logger import logger

class AlertSystem:
    _active_alerts = []
    _halted = False

    @classmethod
    def trigger_alert(cls, message: str, level: str = "ERROR", halt_system: bool = True):
        """Trigger an alert, log it, and optionally halt the system."""
        alert_msg = f"[{level}] {message}"
        if alert_msg not in cls._active_alerts:
            cls._active_alerts.append(alert_msg)
        
        logger.error(f"ALERT TRIGGERED: {alert_msg}")
        
        if halt_system:
            cls._halted = True
            logger.critical("SYSTEM OPERATIONS HALTED due to alert.")

    @classmethod
    def clear_alerts(cls):
        """Clear all active alerts and restore system operation."""
        cls._active_alerts.clear()
        cls._halted = False
        logger.info("All active alerts cleared. System operations resumed.")

    @classmethod
    def is_halted(cls) -> bool:
        """Check if system is halted due to active alerts."""
        return cls._halted

    @classmethod
    def get_active_alerts(cls) -> list:
        """Return the list of active alerts."""
        return cls._active_alerts
