import matplotlib.pyplot as plt
import networkx as nx


class TreeNode(object):
    def __init__(self, name, score, tracklet, duration):
        self.name = name
        self.score = score
        self.id = '{}_{}_{}'.format(duration[0], duration[1], score)
        self.tracklet = tracklet
        self.duration = duration
        self.duration_length = duration[1] - duration[0]
        self.children = list()
        self.parents = list()

    def __repr__(self):
        return self.id


class TrackTree(object):

    def __init__(self, tree_root_name='root', score=0., tracklet=None, duration=None):
        if duration is None:
            duration = [0, 0]
        if tracklet is None:
            tracklet = [0, 0, 0, 0]
        self.count = 0
        self.tree = TreeNode(tree_root_name, score=score, tracklet=tracklet, duration=duration)
        self.id = self.tree.id
        self.if_node_exist = False
        self.search_result_parent = None
        self.search_result_children = []

    def add(self, node, parent=None):
        self.if_node_exist = False
        self.if_node_exist_recursion(
            self.tree, node, search=False, if_del=False)
        if self.if_node_exist:
            # print('Error: Node %s has already existed!' % node.id)
            # print('*' * 30)
            return False
        else:
            if parent is None:
                # if the parent is None, set the root node as default parent
                root_children = self.tree.children
                root_children.append(node)
                self.tree.children = root_children
                # print('Add node:%s sucessfully!' % node.id)
                # print('*' * 30)
                return True
            else:
                # check whether the parent node exists
                self.if_node_exist = False
                self.if_node_exist_recursion(
                    self.tree, parent, search=False, if_del=False)
                if self.if_node_exist:
                    # if parent node exists
                    self.add_recursion(parent.id, node, self.tree)
                    # print('Add node:%s sucessfully!' % node.id)
                    # print('*' * 30)
                    return True
                else:
                    # If parent node doesnt exist
                    # print("Error: Parent node %s doesn't exist!" % parent.id)
                    # print('*' * 30)
                    return False

    def search(self, node):
        self.if_node_exist = False
        self.if_node_exist_recursion(
            self.tree, node, search=True, if_del=False)
        if self.if_node_exist:
            # If exists, return parents node & children
            # print("%s's parent:" % node.id)
            # pt(self.search_result_parent)
            # print("%s's children:" % node.id)
            # pt(self.search_result_children)
            # print('*' * 30)
            return self.search_result_parent, self.search_result_children
        else:
            # If dosent exist
            # print("Error: Node %s doesn't exist!" % node.id)
            # print('*' * 30)
            return None, None

    def delete(self, node):
        self.if_node_exist = False
        self.if_node_exist_recursion(
            self.tree, node, search=False, if_del=True)
        if not self.if_node_exist:
            # print("Error: Node %s doesn't exist!" % node.id)
            # print('*' * 30)
            return False
        else:
            # print('Delete node %s sucessfully!' % node.id)
            # print('*' * 30)
            return True

    def modify(self, node, new_parent=None):
        """Modify parents"""
        self.if_node_exist = False
        self.if_node_exist_recursion(
            self.tree, node, search=False, if_del=False)
        if not self.if_node_exist:
            # print("Error: Node %s doesn't exist!" % node.id)
            # print('*' * 30)
            return False
        else:
            if new_parent is None:
                # If new parent is None, set default root node as parent
                self.if_node_exist = False
                self.if_node_exist_recursion(
                    self.tree, node, search=False, if_del=True)
                root_children = self.tree.children
                root_children.append(node)
                self.tree.children = root_children
                # print('Modify node:%s sucessfully!' % node.id)
                # print('*' * 30)
                return True
            else:
                # Check whether parent exist
                self.if_node_exist = False
                self.if_node_exist_recursion(
                    self.tree, new_parent, search=False, if_del=False)
                if self.if_node_exist:
                    # If parent exist
                    self.if_node_exist = False
                    self.if_node_exist_recursion(
                        self.tree, node, search=False, if_del=True)
                    self.add_recursion(new_parent.id, node, self.tree)
                    # print('Modify node:%s sucessfully!' % node.id)
                    # print('*' * 30)
                    return True
                else:
                    # parent doesnt exist
                    # print("Error: Parent node %s doesn't exist!" %
                    #       new_parent.id)
                    # print('*' * 30)
                    return False

    def show_tree(self):
        G = nx.Graph()
        self.to_graph_recursion(self.tree, G)
        nx.draw_networkx(G, with_labels=True, font_size=10, node_size=5)
        plt.show()

    def to_graph_recursion(self, tree, G):
        G.add_node(tree.id)
        for child in tree.children:
            G.add_nodes_from([tree.id, child.id])
            G.add_edge(tree.id, child.id)
            self.to_graph_recursion(child, G)

    def if_node_exist_recursion(self, tree, node, search, if_del):
        """
        :param tree: check whether exist node tree
        :param node: need 2 check
        :param search: when check the node, whether return parent or all of sons
        :param if_del: when check the node, whether delete it
        :return:
        """
        id = node.id
        if id == self.tree.id:
            self.if_node_exist = True
        if self.if_node_exist:
            return 1
        for child in tree.children:
            if child.id == id:
                self.if_node_exist = True
                if search is True:
                    self.search_result_parent = tree
                    for cchild in child.children:
                        self.search_result_children.append(cchild)
                elif if_del is True:
                    if node in tree.children:
                        tree.children.remove(node)
                        tree.all_nodes.remove(node)
                break
            else:
                self.if_node_exist_recursion(child, node, search, if_del)

    def add_recursion(self, parent, node, tree):
        if parent == tree.id:
            tree.children.append(node)
            return 1
        for child in tree.children:
            if child.id == parent:
                children_list = child.children
                children_list.append(node)
                child.children = children_list
                break
            else:
                self.add_recursion(parent, node, child)


def main():
    T = TrackTree(tree_root_name='adult',
                  score=0.111,
                  tracklet=[1, 1, 1, 1],
                  duration=[0, 30])
    A = TreeNode('adult', 0.1, [2, 2, 2, 2], [0, 30])
    B = TreeNode('adult', 0.2, [3, 3, 3, 3], [0, 30])
    C = TreeNode('adult', 0.3, [4, 4, 4, 4], [15, 45])
    D = TreeNode('adult', 0.4, [5, 5, 5, 5], [30, 60])
    E = TreeNode('adult', 0.5, [6, 6, 6, 6], [30, 60])

    T.add(A)
    T.add(B, A)
    T.add(C, A)
    T.add(C, B)
    T.add(D, C)
    T.add(E, A)

    T.show_tree()


if __name__ == '__main__':
    main()
