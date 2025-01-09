from sklearn.tree import DecisionTreeClassifier
from sklearn.base import ClassifierMixin, RegressorMixin
from sklearn import tree
from openai import OpenAI


class GPTTreeExplainer:
    """
    GPTTreeExplainer class provides an interface for explaining non-interpretable machine learning models using the OpenAI API.

    Attributes:
        api_key (str): The API key for accessing the OpenAI GPT-3 API.
        model (str): The GPT model to be used (default: "gpt-3.5-turbo").
        proxy_tree (sklearn.tree.BaseDecisionTree): A decision tree classifier or regressor used for proxy model training.

    Methods:
        __init__(api_key, model="gpt-3.5-turbo", proxy_tree=DecisionTreeClassifier(max_depth=5)):
            Initializes the GPTTreeExplainer instance with the specified API key, GPT-3 model, and proxy tree.

        fit(X, model):
            Fits the proxy tree on the input data and corresponding model predictions.

        explain(tree_model):
            Generates an explanation for a decision tree using the OpenAI API.

        explain_instance(instance, decision, tree_model):
            Generates an explanation for a specific instance using the OpenAI API.
    """

    api_key = ""
    model = None
    proxy_tree = None

    def __init__(self, api_key, model="gpt-3.5-turbo", proxy_tree=DecisionTreeClassifier(max_depth=5)):
        """
        Initializes the GPTTreeExplainer instance.

        Parameters:
            api_key (str): The API key for accessing the OpenAI API.
            model (str): The GPT model to be used (default: "gpt-3.5-turbo").
            proxy_tree (sklearn.tree.BaseDecisionTree): A decision tree classifier used for proxy model training.
        """

        self.api_key = api_key
        self.model = model
        self.proxy_tree = proxy_tree

    def fit(self, X, model):
        """
       Fits the proxy tree on the input data and corresponding model predictions, and
       returns a textual representation of the fitted decision tree.

       Parameters:
           X (array-like or DataFrame): Input data with features for which explanations are needed.
           model: The machine learning model for which explanations are generated. It should have
                  a `predict` method that takes the input data `X` and returns predictions.

       Returns:
           str: A textual representation of the fitted decision tree, including feature names.
        """

        y = model.predict(X)

        self.proxy_tree.fit(X, y)
        tree_model = tree.export_text(self.proxy_tree, feature_names=X.columns)

        return tree_model

    def explain(self, tree_model, all_data=True):
        """
        Generates an explanation for a decision tree using the OpenAI API.

        Parameters:
            tree_model (str): A textual representation of the fitted decision tree.
            all_data (bool, optional): Flag indicating whether to include information about all data points used in decision-making.
                                       Default is True.

        Returns:
            str: The generated explanation for the decision tree.
        """

        client = OpenAI(api_key=self.api_key)
        prompt = ""

        if isinstance(self.proxy_tree, ClassifierMixin):
            prompt =\
                "You are an expert in machine learning. For each possible final decision(only include each possible decision once), give 5 most important nodes, that influence the decision of the decision tree. Also provide names of all data used in the tree.\n\
                This is the tree:\n" + tree_model + "\n\
                Template for the answer:\n\
                \"In the provided decision tree, the 5 most important nodes for each possible final decision are as follows:\n\n\
                For the decision (first possible final decision):\n\
                 - 1st node\n\
                 - 2nd node\n\
                 - 3rd node\n\
                 - 4th node\n\
                 - 5th node\n\n\
                For the decision (second possible final decision):\n\
                 - 1st node\n\
                 - 2nd node\n\
                 - 3rd node\n\
                 - 4th node\n\
                 - 5th node\n\""

        elif isinstance(self.proxy_tree, RegressorMixin):
            prompt = \
                "You are an expert in machine learning. For the high, middle and low values, give 5 most important nodes, that influence the decision of the decision tree.\n\
                This is the tree:\n" + tree_model + "\n\
                Template for the answer:\n\
                \"In the provided decision tree, the 5 most important nodes for each possible final decision are as follows:\n\n\
                For the decision of a low value (minimum value - maximum value):\n\
                 - 1st node\n\
                 - 2nd node\n\
                 - 3rd node\n\
                 - 4th node\n\
                 - 5th node\n\n\
                For the decision of a medium value (*provide lowest and highest decisions of a medium value:\n\
                 - 1st node\n\
                 - 2nd node\n\
                 - 3rd node\n\
                 - 4th node\n\
                 - 5th node\n\n\
                 For the decision of a high value (*provide lowest and highest decisions of a high value):\n\
                 - 1st node\n\
                 - 2nd node\n\
                 - 3rd node\n\
                 - 4th node\n\
                 - 5th node\n\""

        if all_data:
            prompt = prompt + "\"The rest of the data, that influences the decision:\n\
                (provide names of all data points that are used in the decision making, but have not yet been mentioned)\""

        answer = client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            stream=False,
            max_tokens=500
        )

        return answer.choices[0].message.content

    def explain_instance(self, instance, decision, tree_model, all_data=True):
        """
            Generates an explanation for a specific instance and decision in a decision tree
            using the OpenAI API.

            Parameters:
                instance (array-like or DataFrame): A textual representation of the data instance for which an explanation is needed.
                decision (str): The decision or outcome for which the explanation is generated.
                tree_model (str): A textual representation of the fitted decision tree.
                all_data (bool, optional): Flag indicating whether to include information about all data points used in decision-making.
                                       Default is True.

            Returns:
                str: The generated explanation for the specific instance and decision in the decision tree.
            """

        client = OpenAI(api_key=self.api_key)

        prompt = \
            "You are an expert in machine learning. For the decision that was given in this instance, give 5 most important nodes, that influence the decision of the decision tree. Also provide names of all data used for the decision.\n\
            This is the instance:\n" + instance + "\n\
            This Is the decision:" + decision + "\n\
            This is the tree:\n" + tree_model + "\n\
            Template for the answer:\n\
            \"In the provided decision tree, the 5 most important nodes for the given decision decision are as follows:\n\n\
            For the decision (the decision provided):\n\
            - 1st node\n\
            - 2nd node\n\
            - 3rd node\n\
            - 4th node\n\
            - 5th node\n\""

        if all_data:
            prompt = prompt + "\"The rest of the data, that influences the decision:\n\
                (provide names of all data points that are used in the decision making, but have not yet been mentioned)\""

        answer = client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            stream=False,
            max_tokens=500
        )

        return answer.choices[0].message.content
