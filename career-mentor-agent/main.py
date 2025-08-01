import os
from openai import OpenAI
from typing import Dict, List, Optional

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY", "sk-test-key"))  # Fallback for testing

# ======================
# Tool Implementation
# ======================
def get_career_roadmap(career: str) -> str:
    """Tool: Returns skill roadmap for a specific career"""
    roadmaps = {
        "Software Engineer": [
            "1. Programming fundamentals (Python/Java)",
            "2. Data structures & algorithms",
            "3. Version control (Git)",
            "4. Software design patterns",
            "5. Cloud computing basics",
            "6. CI/CD pipelines"
        ],
        "Web Developer": [
            "1. HTML, CSS, JavaScript fundamentals",
            "2. Git & GitHub version control",
            "3. Frontend Frameworks (React, Next.js)",
            "4. Backend Development (Node.js, Express)",
            "5. Databases (MongoDB, PostgreSQL)",
            "6. Deployment (Vercel, Netlify)"
        ],
        "Data Scientist": [
            "1. Python, Pandas, NumPy",
            "2. Data Visualization (Matplotlib, Seaborn)",
            "3. Machine Learning (scikit-learn, TensorFlow)",
            "4. SQL for data querying",
            "5. Statistical Analysis",
            "6. Model Deployment"
        ],
        "UX Designer": [
            "1. User Research Methods",
            "2. Wireframing & Prototyping",
            "3. UI Design Principles",
            "4. Design Tools (Figma, Sketch)",
            "5. Usability Testing",
            "6. Interaction Design"
        ],
        "Cybersecurity Analyst": [
            "1. Network Security Fundamentals",
            "2. Security Protocols & Cryptography",
            "3. Ethical Hacking Techniques",
            "4. Security Compliance Standards",
            "5. Incident Response",
            "6. Threat Intelligence"
        ]
    }
    
    # Case-insensitive matching
    normalized_career = career.lower()
    for key in roadmaps:
        if key.lower() == normalized_career:
            return f"📚 {key} Skill Roadmap:\n" + "\n".join(roadmaps[key])
    
    return f"⚠️ No roadmap available for {career}"

# ======================
# Agent System Simulation
# ======================
class CareerMentor:
    def __init__(self):
        self.agents = {
            "career": self._career_agent,
            "skill": self._skill_agent,
            "job": self._job_agent
        }
        self.current_agent = "career"
        self.context = {"career": None, "recommendations": []}
        self.awaiting_selection = False
    
    def _career_agent(self, user_input: str) -> str:
        """Recommends careers based on interests"""
        # If we're waiting for career selection
        if self.awaiting_selection:
            normalized_input = user_input.lower()
            
            # Find matching career from recommendations
            for career in self.context["recommendations"]:
                if career.lower() in normalized_input:
                    self.context["career"] = career
                    self.awaiting_selection = False
                    self.current_agent = "skill"
                    return f"🚀 Great choice! Exploring {career} now..."
            
            # If no match found
            return (
                "⚠️ Please select a career from the recommendations:\n" +
                "\n".join(f"- {c}" for c in self.context["recommendations"]) +
                "\n\nOr describe your interests for new suggestions."
            )
        
        # First-time recommendation
        interests = user_input.lower()
        
        if "data" in interests or "stat" in interests or "analy" in interests:
            careers = ["Data Scientist", "Data Analyst"]
        elif "design" in interests or "ui" in interests or "ux" in interests or "interface" in interests:
            careers = ["UX Designer", "Product Designer"]
        elif "security" in interests or "cyber" in interests or "hack" in interests:
            careers = ["Cybersecurity Analyst", "Security Engineer"]
        elif "web" in interests or "front" in interests or "back" in interests:
            careers = ["Web Developer", "Frontend Developer", "Backend Developer"]
        else:
            careers = ["Software Engineer", "Full Stack Developer"]
        
        self.context["recommendations"] = careers
        self.awaiting_selection = True
        
        return (
            f"🔍 Based on your interests, I recommend:\n" +
            "\n".join(f"- {c}" for c in careers) +
            "\n\nPlease type the name of a career to explore further"
        )
    
    def _skill_agent(self, _: Optional[str] = None) -> str:
        """Shows skill roadmap for selected career"""
        career = self.context["career"]
        roadmap = get_career_roadmap(career)
        return f"{roadmap}\n\nType 'jobs' to see career opportunities"

    def _job_agent(self, _: Optional[str] = None) -> str:
        """Provides job market insights"""
        career = self.context["career"]
        job_data = {
            "Software Engineer": {
                "roles": ["Frontend Developer", "Backend Developer", "Full Stack Engineer", "DevOps Engineer"],
                "salary": "$80k-$160k",
                "demand": "High across all industries"
            },
            "Web Developer": {
                "roles": ["Frontend Developer", "Backend Developer", "Full Stack Developer"],
                "salary": "$70k-$140k",
                "demand": "Strong in tech and e-commerce"
            },
            "Data Scientist": {
                "roles": ["Data Analyst", "Machine Learning Engineer", "Business Intelligence Specialist"],
                "salary": "$90k-$170k", 
                "demand": "Growing in tech/finance/healthcare"
            },
            "UX Designer": {
                "roles": ["UI Designer", "UX Researcher", "Product Designer"],
                "salary": "$75k-$140k",
                "demand": "Strong in tech/e-commerce"
            },
            "Cybersecurity Analyst": {
                "roles": ["Security Analyst", "Penetration Tester", "Security Architect"],
                "salary": "$90k-$150k",
                "demand": "High in finance/government/tech"
            }
        }
        
        # Find best match (case-insensitive)
        normalized_career = career.lower()
        matched_career = next(
            (key for key in job_data if key.lower() == normalized_career), 
            "Technology Professional"
        )
        
        data = job_data.get(matched_career, {
            "roles": ["Various technical roles"],
            "salary": "$70k-$160k (varies by experience)",
            "demand": "Strong in technology sector"
        })
        
        response = (
            f"💼 Job Market for {matched_career}:\n" +
            f"Common Roles: {', '.join(data['roles'])}\n" +
            f"Salary Range: {data['salary']}\n" +
            f"Market Demand: {data['demand']}\n\n" +
            "Type 'new' to explore another career"
        )
        return response
    
    def _handle_agent_handoff(self, user_input: str) -> None:
        """Manages transitions between agents"""
        user_input = user_input.lower()
        
        # Handle career selection
        if self.current_agent == "career" and self.awaiting_selection:
            for career in self.context["recommendations"]:
                if career.lower() in user_input:
                    self.context["career"] = career
                    self.awaiting_selection = False
                    self.current_agent = "skill"
                    return
        
        # Normal handoff triggers
        if self.current_agent == "skill" and ("job" in user_input or "opportunit" in user_input):
            self.current_agent = "job"
        elif self.current_agent == "job" and "new" in user_input:
            self.current_agent = "career"
            self.context["career"] = None
            self.awaiting_selection = False
    
    def run(self, user_input: str) -> str:
        """Main execution flow"""
        self._handle_agent_handoff(user_input)
        agent = self.agents[self.current_agent]
        
        # Handle input for current agent
        if self.current_agent == "career":
            return agent(user_input)
        return agent()

# ======================
# Main Application
# ======================
if __name__ == "__main__":
    mentor = CareerMentor()
    
    print("🌟 Career Mentor Agent 🌟")
    print("Describe your interests (e.g. 'I enjoy problem solving with data')")
    print("Type 'exit' to quit\n")
    
    while True:
        user_input = input("You: ").strip()
        
        if user_input.lower() in ["exit", "quit"]:
            print("Goodbye! 👋")
            break
        
        response = mentor.run(user_input)
        print(f"\nMentor: {response}\n")