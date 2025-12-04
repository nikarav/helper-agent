import time
from dataclasses import dataclass

from helper_agent.utilities.logger import get_logger

logger = get_logger("helper_agent")


@dataclass
class RateLimiter:
    """Rate limiter for API calls with RPM and TPM limits."""

    rpm: int = 60  # Requests per minute
    tpm: int = 100000  # Tokens per minute

    def __post_init__(self) -> None:
        self._request_times: list[float] = []
        self._token_counts: list[tuple[float, int]] = []

    def _cleanup_old_entries(self, current_time: float) -> None:
        """
        Remove entries older than 1 minute.

        :param current_time: Current time
        """
        cutoff = current_time - 60.0

        # clean request times
        self._request_times = [t for t in self._request_times if t > cutoff]

        # clean token counts
        self._token_counts = [(t, c) for t, c in self._token_counts if t > cutoff]

    def _current_rpm(self) -> int:
        """Get current requests in the last minute."""
        return len(self._request_times)

    def _current_tpm(self) -> int:
        """Get current tokens in the last minute."""
        return sum(c for _, c in self._token_counts)

    def wait_if_needed(self, estimated_tokens: int) -> None:
        """
        Wait if rate limits would be exceeded.

        :param estimated_tokens: Estimated tokens for the upcoming request
        """
        while True:
            current_time = time.time()
            self._cleanup_old_entries(current_time)

            # check RPM
            if self._current_rpm() >= self.rpm:
                oldest = (
                    min(self._request_times) if self._request_times else current_time
                )
                wait_time = 60.0 - (current_time - oldest) + 0.1
                if wait_time > 0:
                    logger.debug(f"RPM limit reached, waiting {wait_time:.1f}s")
                    time.sleep(wait_time)
                    continue

            # check TPM
            if self._current_tpm() + estimated_tokens > self.tpm:
                oldest = (
                    min(t for t, _ in self._token_counts)
                    if self._token_counts
                    else current_time
                )
                wait_time = 60.0 - (current_time - oldest) + 0.1
                if wait_time > 0:
                    logger.debug(f"TPM limit reached, waiting {wait_time:.1f}s")
                    time.sleep(wait_time)
                    continue

            break

    def record_request(self, token_count: int) -> None:
        """
        Record a completed request.

        :param token_count: Number of tokens used
        """
        current_time = time.time()
        self._request_times.append(current_time)
        self._token_counts.append((current_time, token_count))
