import sys
from PyQt5.QtWidgets import (
    QApplication,
    QWidget,
    QVBoxLayout,
    QTextEdit,
    QLineEdit,
    QPushButton,
)
from llama import get_response_from_llama
from sentiment_analysis import sentiment_analysis
from predict_loan_status import model_pred_status, input_info

class ChatBot(QWidget):
    def __init__(self):
        """Initializes the ChatBot application."""
        super().__init__()
        self.setWindowTitle("Loan Enquiry and Credit Risk Analysis Chatbot")
        self.resize(500, 600)  # Set window size to 500x600 pixels
        self.setup_ui()
        # Initialize the stage of the conversation
        self.stage = 0
        self.reason_anal = ''  # Variable to store the analyzed reason for loan application
        self.pred_input = []  # List to store user inputs for loan prediction
        self.pred_loan_status = ''  # Variable to store the predicted loan status

    def setup_ui(self):
        """Sets up the user interface for the chatbot."""
        self.layout = QVBoxLayout()

        self.chat_display = QTextEdit()
        self.chat_display.setReadOnly(True)
        self.layout.addWidget(self.chat_display)

        # Add the greeting message here
        self.chat_display.append("Bot: Hello! How are you today?")
        self.chat_display.append("Bot: I'm your consultant on loan enquiry and credit risk analysis.")
        self.chat_display.append("Bot: Let's start by discussing your reasons for loan application.")

        self.input_field = QLineEdit()
        self.layout.addWidget(self.input_field)

        self.input_field.returnPressed.connect(self.handle_message)  # Bind Enter key to send
        self.send_button = QPushButton("Send")
        self.send_button.clicked.connect(self.handle_message)
        self.layout.addWidget(self.send_button)

        self.setLayout(self.layout)

    def handle_message(self):
        """Handles the user's input message and generates a response."""
        user_text = self.input_field.text()
        if user_text:
            self.chat_display.append(f"You: {user_text}")
            bot_response = self.get_response(user_text)
            if isinstance(bot_response, list):
                for response in bot_response:
                    self.chat_display.append(response)
            else:
                self.chat_display.append(f"Bot: {bot_response}")
            self.input_field.clear()

    def get_response(self, message):
        """
        Generates a response based on the current stage of the conversation.
        Stages:
        0: Analyze the reason for loan application
        1: Handle credit grade input
        2: Handle employer's name input
        3: Handle employment length input
        4: Handle home ownership status input
        5: Handle annual income input
        6: Handle verification status input
        7: Handle debt-to-income ratio input
        8: Handle number of delinquencies input
        9: Handle total current balance input
        10: Handle total revolving high credit/credit limit input
        11: Handle final report generation
        """
        if self.stage == 0:
            # Analyze the reason for loan application
            self.reason_anal = sentiment_analysis(message) 
            ret_message = []
            ret_message.append('<span style="color: yellow;">Bot: Your reason is ' + self.reason_anal + ".")
            ret_message.append("Bot: Let's move on to other information.")
            ret_message.append("Bot: Please input your grade(From A to G).")
            self.stage = 1
            return ret_message
        
        elif self.stage == 1:
            # Handle grade input
            grade = message.upper()
            ret_message = ""
            if grade not in ['A', 'B', 'C', 'D', 'E', 'F', 'G']:
                ret_message += "Invalid Grade!"
                ret_message += '\n'
                ret_message += "Bot: Please Re-input your grade(From A to G)."
                return ret_message
            else:
                self.pred_input.append(grade)
                ret_message += "Please input your employer's name."
                self.stage = 2
                return ret_message
            
        elif self.stage == 2:
            # Handle employer's name input
            emp_title = message
            self.pred_input.append(emp_title)
            ret_message = "Please input your employment length.(in years)"
            self.stage = 3
            return ret_message
        
        elif self.stage == 3:
            # Handle employment length input
            emp_length = message
            # Check if the input is a number
            if not emp_length.isdigit():
                ret_message = "Employment Length should be a number!"
                ret_message += '\n'
                ret_message += "Bot: Please Re-input your employment length.(in years)"
                return ret_message
            else:
                years = int(emp_length)
                if years < 0:
                    ret_message = "Employment Length should be a positive number!"
                    ret_message += '\n'
                    ret_message += "Bot: Please Re-input your employment length.(in years)"
                    return ret_message
                elif years == 0:
                    emp_length = "0 years"
                elif years > 10:
                    emp_length = "10+ years"
                else:
                    emp_length = str(years) + " years"

                self.pred_input.append(emp_length)
                ret_message = "Please input your home ownership status.(O: Own, M: Mortgage, R: Rent)"
                self.stage = 4
                return ret_message
            
        elif self.stage == 4:
            # Handle home ownership status input
            home_ownership = message.upper()
            if home_ownership not in ['O', 'M', 'R']:
                ret_message = "Invalid Home Ownership Status!"
                ret_message += '\n'
                ret_message += "Bot: Please Re-input your home ownership status.(O: Own, M: Mortgage, R: Rent)"
                return ret_message
            else:
                if home_ownership == 'O':
                    home_ownership = "OWN"
                elif home_ownership == 'M':
                    home_ownership = "MORTGAGE"
                else:
                    home_ownership = "RENT"
                self.pred_input.append(home_ownership)
                ret_message = "Please input your annual income."
                self.stage = 5
                return ret_message
            
        elif self.stage == 5:
            # Handle annual income input
            annual_inc = message
            if not annual_inc.isdigit():
                ret_message = "Annual Income should be a number!"
                ret_message += '\n'
                ret_message += "Bot: Please Re-input your annual income."
                return ret_message
            else:
                self.pred_input.append(annual_inc)
                ret_message = "Please input your verification status.(V: Verified, N: Not Verified, S: Source Verified)"
                self.stage = 6
                return ret_message
        
        elif self.stage == 6:
            # Handle verification status input
            verification_status = message.upper()
            if verification_status not in ['V', 'N', 'S']:
                ret_message = "Invalid Verification Status!"
                ret_message += '\n'
                ret_message += "Bot: Please Re-input your verification status.(V: Verified, N: Not Verified)"
                return ret_message
            else:
                if verification_status == 'V':
                    verification_status = "Verified"
                elif verification_status == 'S':
                    verification_status = "Source Verified"
                else:
                    verification_status = "Not Verified"
                self.pred_input.append(verification_status)
                ret_message = "Please input your debt-to-income ratio."
                self.stage = 7
                return ret_message
            
        elif self.stage == 7:
            # Handle debt-to-income ratio input
            dti = message
            try:
                dti = float(dti)
            except ValueError:
                ret_message = "Debt-to-Income Ratio should be a number!"
                ret_message += '\n'
                ret_message += "Bot: Please Re-input your debt-to-income ratio."
                return ret_message
            else:
                if dti < 0:
                    ret_message = "Debt-to-Income Ratio should be a positive number!"
                    ret_message += '\n'
                    ret_message += "Bot: Please Re-input your debt-to-income ratio."
                    return ret_message
                self.pred_input.append(str(dti))
                ret_message = "Please input your number of delinquencies in the past 2 years."
                self.stage = 8
                return ret_message

        elif self.stage == 8:
            # Handle number of delinquencies input
            delinq_2yrs = message
            if not delinq_2yrs.isdigit():
                ret_message = "Number of Delinquencies should be a number!"
                ret_message += '\n'
                ret_message += "Bot: Please Re-input your number of delinquencies."
                return ret_message
            else:
                self.pred_input.append(delinq_2yrs)
                ret_message = "Please input your total current balance."
                self.stage = 9
                return ret_message
        
        elif self.stage == 9:
            # Handle total current balance input
            tot_cur_bal = message
            if not tot_cur_bal.isdigit():
                ret_message = "Total Current Balance should be a number!"
                ret_message += '\n'
                ret_message += "Bot: Please Re-input your total current balance."
                return ret_message
            else:
                self.pred_input.append(tot_cur_bal)
                ret_message = "Please input your total revolving high credit/credit limit."
                self.stage = 10
                return ret_message
        
        elif self.stage == 10:
            # Handle total revolving high credit/credit limit input
            total_rev_hi_lim = message
            if not total_rev_hi_lim.isdigit():
                ret_message = "Total Revolving High Credit/Credit Limit should be a number!"
                ret_message += '\n'
                ret_message += "Bot: Please Re-input your total revolving high credit/credit limit."
                return ret_message
            else:
                self.pred_input.append(total_rev_hi_lim)
                ret_message = []
                ret_message.append("I'm analyzing your loan status.")
                ret_message.append("Bot: Please wait for a moment.")
                self.pred_loan_status = model_pred_status(input_info(self.pred_input))
                ret_message.append('<span style="color: yellow;">Bot: Your credit risk is ' + self.pred_loan_status + ".")
                ret_message.append("Bot: I'm ready to give you a final report on loan and credit risk.")
                ret_message.append("Bot: Please type 'report' to get the report.")
                self.stage = 11
                return ret_message
            
        elif self.stage == 11:
            # Handle final report generation
            if message != 'report':
                ret_message = "Invalid Input!"
                ret_message += '\n'
                ret_message += "Bot: Please type 'report' to get the report."
                return ret_message
            else:
                self.chat_display.append("Bot: Here's your final report on loan and credit risk.")
                self.stage = 0
                return get_response_from_llama(message, self.reason_anal, self.pred_loan_status)


if __name__ == "__main__":
    # Main entry point for the application
    app = QApplication(sys.argv)
    window = ChatBot()
    window.show()
    sys.exit(app.exec_())