import re
import html
import unicodedata
import string
import logging


# set up logging
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.WARNING)

if not logger.handlers:
    ch = logging.FileHandler("sanitizer.log")
    ch.setLevel(logging.WARNING)

    formatter = logging.Formatter("%(asctime)s - %(name)s - %(message)s")
    ch.setFormatter(formatter)

    logger.addHandler(ch)
            
class InputSanitizer:
    
    SUSPICIOUS_PATTERNS = [
        r"ignore\s+previous\s+instructions",
        r"you\s+are\s+(a|an)?\s*(helpful)?\s*assistant",
        r"pretend\s+to\s+be",
        r"repeat\s+after\s+me",
        r"disregard\s+(all|any)?\s*(prior|previous)?\s*instructions",
        r"follow\s+these\s+instructions",
        r"act\s+as\s+(a|an)?",
        r"email\s+body.*(do\s+the\s+following|follow\s+these\s+steps)",
        r"meeting\s+notes?.*(must\s+respond|override\s+previous\s+instructions)",
        r"task\s+(body|description).*(ignore\s+security|bypass\s+rules|change\s+system\s+behavior)"
    ]
    
    # idk what the input is !! check!!
    @staticmethod
    def normalize_text(text: str) -> str:
        """
        Converts stylized Unicode to plain ASCII equivalents
        """        
        return unicodedata.normalize("NFKC", text)

    @staticmethod
    def escape_html_tags(text: str) -> str:
        """
        Remove html tags
        """
        return html.escape(text)
        
    def remove_suspicious_patterns(self, text: str) -> str:
        """
        Redacts suspicious phrases using regex pattern matching.
        """
        normalized = self.normalize_text(text)
        normalized = self.escape_html_tags(normalized)

        count_of_detected_pattern = 0

        for pattern in self.SUSPICIOUS_PATTERNS:
            if re.search(pattern, normalized, flags=re.IGNORECASE):
                logger.warning(f"Suspicious pattern detected and redacted: {pattern}")
                normalized = re.sub(pattern, "[REDACTED]", normalized, flags=re.IGNORECASE)
                count_of_detected_pattern += 1

        logger.warning(f"Total detected suspicious patterns: {count_of_detected_pattern}")

        return normalized