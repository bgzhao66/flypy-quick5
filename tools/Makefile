# generate a list of tables to be converted from prefix 'pinvin_simp_ext' and suffix '2' to '7'
# and the primary table 'pinvin_simp'

PRIMARY_NAME = flypy_quick5
INDEXES = $(shell echo A B)
# Loop over the indexes and generate the table names
# The table names are the primary table name with the index appended
INPUT_TABLES = $(foreach index,$(INDEXES),$(PRIMARY_NAME)_ext$(index))

.SILENT:
all: primary_table predef_table abbrev_table extra_tables
	echo "All tables converted"

primary_table:
	echo "Converting to table $(PRIMARY_NAME)"
	python3 ./convert_to_flypy_quick5.py --chinese_code --name $(PRIMARY_NAME) --input_tables $(PRIMARY_NAME)_abbrev $(PRIMARY_NAME)_ext $(INPUT_TABLES) > ../$(PRIMARY_NAME).dict.yaml

predef_table:
	echo "Converting to table $(PRIMARY_NAME)_ext"
	python3 ./convert_to_flypy_quick5.py --name $(PRIMARY_NAME)_ext --pinyin_phrase > ../$(PRIMARY_NAME)_ext.dict.yaml

abbrev_table:
	echo "Converting to abbreviated table $(ABBREV_TABLE)"
	python3 ./convert_to_flypy_quick5.py --abbreviate --name $(PRIMARY_NAME)_abbrev > ../$(PRIMARY_NAME)_abbrev.dict.yaml

extra_tables:
	for table in $(shell echo $(INDEXES)); do \
		echo "Converting to table $(PRIMARY_NAME)_ext$${table}"; \
		python3 ./convert_to_flypy_quick5.py words_$${table}.txt --difference --name $(PRIMARY_NAME)_ext$${table}  > ../$(PRIMARY_NAME)_ext$${table}.dict.yaml; \
	done

.PHONY: clean
clean:
	rm -f ../$(PRIMARY_NAME).dict.yaml ../$(PRIMARY_NAME)_ext*.dict.yaml
