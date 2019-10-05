from DoublyLinkedList import DoublyLinkedList
linked_list_sent1 = DoublyLinkedList()
linked_list_sent2 = DoublyLinkedList()

sent1 = 'great food and crunchy taste'
sent2 = 'food was delicious and would recommend this to everyone'

for word in sent1.split()[::-1]:
    linked_list_sent1.insert_at_start(word)

for word in sent2.split()[::-1]:
    linked_list_sent2.insert_at_start(word)


linked_list_sent1.traverse_list()
print('***************')
linked_list_sent2.traverse_list()

list_sent = list()
list_sent.append(sent1)
list_sent.append(sent2)
common_words = [word for word in sent1.split() if word in sent2.split()]
print(common_words)


def traverse_match(primary_list, secondary_list, match_string):
    if linked_list_sent1.start_node is None or linked_list_sent2.start_node is None:
        print('The lists are empty')
    else:
        for word in common_words:
            word_list = list()
            m = primary_list.start_node
            while m is not None:
                if m.item != word:
                    word_list.append(m.item)
                    m = m.nref
                else:
                    word_list.append(m.item)
                    n = secondary_list.start_node
                    while n is not None:
                        if n.item == word:
                            n = n.nref
                            word_list.append(n.item)
                            n = n.nref
                            while n is not None:
                                word_list.append(n.item)
                                n = n.nref
                        else:
                            n = n.nref
                            continue
                    break
            match_string.append([' '+word for word in word_list])

match_string = list()
traverse_match(linked_list_sent1, linked_list_sent2, match_string)
traverse_match(linked_list_sent2, linked_list_sent1, match_string)
print(match_string)
for sentence in match_string:
    s = [str(word) for word in sentence]
    print(" ".join(s))
