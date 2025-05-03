from marker.converters.pdf import PdfConverter
from marker.models import create_model_dict
from marker.config.parser import ConfigParser
from marker.converters.table import TableConverter

filepath = "../ragFile/채용 업무처리지침.pdf"

config = {
    "output_format": "json",
    "ADDITIONAL_KEY": "VALUE"
}
config_parser = ConfigParser(config)

converter = PdfConverter(
    config=config_parser.generate_config_dict(),
    artifact_dict=create_model_dict(),
    processor_list=config_parser.get_processors(),
    renderer=config_parser.get_renderer(),
    llm_service=config_parser.get_llm_service()
)
rendered = converter(filepath)
text, _, images = text_from_rendered(rendered)

print("1111:\n", text)
print("1111:\n", images)



converter2 = TableConverter(
    artifact_dict=create_model_dict(),
)
rendered2 = converter2(filepath)
text, _, images = text_from_rendered(rendered2)

print("222:\n", text)
print("222:\n", images)