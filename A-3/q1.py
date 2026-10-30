from visual_automata.fa.dfa import VisualDFA
import string

lowercase = set(string.ascii_lowercase)
uppercase = set(string.ascii_uppercase)
digits = set(string.digits)
special = set(" _.,!?@#$%^&*()-+=[]{}:;'\"/\\|<>`~")

alphabet = lowercase | uppercase | digits | special

transitions = {
    "q0": {},
    "q1": {},
    "q2": {}
}

for ch in alphabet:
    if ch in lowercase:
        transitions["q0"][ch] = "q1"
        transitions["q1"][ch] = "q1"
    else:
        transitions["q0"][ch] = "q2"
        transitions["q1"][ch] = "q2"

    transitions["q2"][ch] = "q2"

dfa = VisualDFA(
    states={"q0", "q1", "q2"},
    input_symbols=alphabet,
    transitions=transitions,
    initial_state="q0",
    final_states={"q1"}
)

print("Transition Table:")
print(dfa.table)

word = input("Enter a word: ")

print("\nResult:")
dfa.input_check(word)

dfa.show_diagram(word)