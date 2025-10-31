PROJECTS := softmax vector-addition

.PHONY: default run clean $(PROJECTS) $(addprefix clean-,$(PROJECTS))

default: run clean

run: $(PROJECTS)

clean: $(addprefix clean-,$(PROJECTS))

$(PROJECTS):
	@echo "Building and running $@..."
	@cd $@ && nvcc main.cu solution.cu -o ${@}_test && ./${@}_test

$(addprefix clean-,$(PROJECTS)):
	@echo "Cleaning $(subst clean-,,$@)..."
	@cd $(subst clean-,,$@) && rm -f $(subst clean-,,$@)_test