
from ui import run_ui
import warnings

# Suppress specific UserWarning from torchaudio
warnings.filterwarnings(
    "ignore",
    message=".*torchaudio._backend.list_audio_backends has been deprecated.*",
    category=UserWarning
)

# Suppress specific FutureWarning from torch.cuda.amp
warnings.filterwarnings(
    "ignore",
    message=".*torch.cuda.amp.custom_fwd.*is deprecated.*",
    category=FutureWarning
)


if __name__ == "__main__":
    run_ui()