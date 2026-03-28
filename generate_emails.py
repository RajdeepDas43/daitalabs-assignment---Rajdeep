#!/usr/bin/env python3
"""
Generate 100 synthetic emails for RAG evaluation
"""
import os
import random
from pathlib import Path

# Generate 200 unique names
first_names = [
    "Alex", "Jordan", "Taylor", "Morgan", "Casey", "Riley", "Avery", "Quinn",
    "Blake", "Cameron", "Dakota", "Emery", "Finley", "Harper", "Hayden", "Jamie",
    "Kai", "Logan", "Marley", "Noah", "Parker", "Reese", "River", "Rowan",
    "Sage", "Skyler", "Tatum", "Winter", "Zephyr", "Adrian", "Blair", "Carter",
    "Drew", "Ellis", "Gray", "Hollis", "Indigo", "Jules", "Kendall", "Lane",
    "Micah", "Nico", "Ocean", "Peyton", "Quincy", "Remy", "Sasha", "Tristan",
    "Val", "Wren", "Aaron", "Benjamin", "Christopher", "Daniel", "Ethan", "Frank",
    "George", "Henry", "Isaac", "James", "Kevin", "Liam", "Michael", "Nathan",
    "Oliver", "Patrick", "Quinn", "Robert", "Samuel", "Thomas", "Victor", "William",
    "Xavier", "Yusuf", "Zachary", "Alice", "Beth", "Clara", "Diana", "Emma",
    "Fiona", "Grace", "Hannah", "Isabella", "Julia", "Kate", "Lily", "Maya",
    "Nora", "Olivia", "Penelope", "Quinn", "Rachel", "Sophia", "Tara", "Uma",
    "Victoria", "Wendy", "Xara", "Yara", "Zoe", "Adam", "Brian", "Charles",
    "David", "Edward", "Felix", "Gavin", "Harold", "Ivan", "Jack", "Kyle",
    "Luke", "Mark", "Neil", "Oscar", "Paul", "Ryan", "Steve", "Tyler",
    "Ulysses", "Vincent", "Walter", "Xander", "Yves", "Zane", "Amanda", "Bella",
    "Catherine", "Diana", "Elena", "Faith", "Gina", "Helen", "Iris", "Jane",
    "Kara", "Laura", "Maria", "Nina", "Opal", "Paula", "Rita", "Sara",
    "Tina", "Una", "Vera", "Wanda", "Xena", "Yvonne", "Zara", "Andrew",
    "Brandon", "Caleb", "Derek", "Evan", "Felix", "Graham", "Hugo", "Ian",
    "Jake", "Kane", "Leo", "Max", "Nick", "Owen", "Peter", "Quinn",
    "Rex", "Sean", "Tom", "Uri", "Vince", "Wade", "Xavi", "Yan",
    "Zack", "Anna", "Bella", "Cora", "Dana", "Eva", "Faye", "Gia",
    "Hope", "Ivy", "Joy", "Kira", "Luna", "Mia", "Nia", "Ora"
]

last_names = [
    "Smith", "Johnson", "Williams", "Brown", "Jones", "Garcia", "Miller", "Davis",
    "Rodriguez", "Martinez", "Hernandez", "Lopez", "Wilson", "Anderson", "Thomas", "Taylor",
    "Moore", "Jackson", "Martin", "Lee", "Thompson", "White", "Harris", "Sanchez",
    "Clark", "Ramirez", "Lewis", "Robinson", "Walker", "Young", "Allen", "King",
    "Wright", "Scott", "Torres", "Nguyen", "Hill", "Flores", "Green", "Adams",
    "Nelson", "Baker", "Hall", "Rivera", "Campbell", "Mitchell", "Carter", "Roberts",
    "Gomez", "Phillips", "Evans", "Turner", "Diaz", "Parker", "Cruz", "Edwards",
    "Collins", "Reyes", "Stewart", "Morris", "Morales", "Murphy", "Cook", "Rogers",
    "Gutierrez", "Ortiz", "Morgan", "Cooper", "Peterson", "Bailey", "Reed", "Kelly",
    "Howard", "Ramos", "Kim", "Cox", "Ward", "Richardson", "Watson", "Brooks",
    "Chavez", "Wood", "James", "Bennett", "Gray", "Mendoza", "Ruiz", "Hughes",
    "Price", "Alvarez", "Castillo", "Sanders", "Patel", "Myers", "Long", "Ross",
    "Foster", "Jimenez", "Powell", "Jenkins", "Perry", "Russell", "Sullivan", "Bell",
    "Coleman", "Butler", "Henderson", "Barnes", "Gonzales", "Fisher", "Vasquez", "Simmons",
    "Romero", "Jordan", "Patterson", "Alexander", "Hamilton", "Graham", "Reynolds", "Griffin",
    "Wallace", "Moreno", "West", "Cole", "Hayes", "Bryant", "Herrera", "Gibson",
    "Ellis", "Tran", "Medina", "Aguilar", "Stevens", "Murray", "Ford", "Castro",
    "Marshall", "Owens", "Harrison", "Fernandez", "Mcdonald", "Woods", "Washington", "Kennedy"
]

# Generate 200 unique people
people = []
for _ in range(200):
    first = random.choice(first_names)
    last = random.choice(last_names)
    email = f"{first.lower()}.{last.lower()}@{random.choice(['company.com', 'tech.io', 'business.net', 'corp.org', 'enterprise.com'])}"
    people.append({
        'name': f"{first} {last}",
        'email': email
    })

# Email templates with diverse topics
topics = [
    ("Project Update", "I wanted to provide you with an update on the current project status. We have made significant progress over the past few weeks and I believe it's important to keep everyone informed about the developments. The team has been working diligently to meet the deadlines and ensure quality standards are maintained throughout the process. We encountered some challenges along the way, particularly with the integration phase, but we've managed to resolve most of the issues through collaborative problem-solving sessions. The next milestone is approaching quickly, and I'm confident we'll be able to deliver on time. Please let me know if you have any questions or concerns about the current progress."),
    
    ("Meeting Request", "I hope this email finds you well. I would like to schedule a meeting to discuss the upcoming quarter's strategy and review our performance metrics. This meeting is crucial for aligning our team's objectives and ensuring we're all on the same page regarding the next steps. I've prepared a comprehensive presentation that covers our key achievements, current challenges, and proposed solutions. We should also allocate time to discuss resource allocation and budget considerations for the upcoming projects. I'm flexible with timing, so please let me know what works best for your schedule. I'm looking forward to our productive discussion."),
    
    ("Budget Approval", "I'm writing to request approval for the proposed budget allocation for the next fiscal year. After careful analysis and consultation with the finance team, we've identified several areas where strategic investment would yield significant returns. The budget proposal includes provisions for new equipment, additional staffing, and enhanced training programs. We've also factored in contingency funds to handle unexpected expenses that may arise during the implementation phase. I've attached a detailed breakdown of all expenses and their justifications. I believe this investment is essential for maintaining our competitive edge and ensuring continued growth. Please review the proposal and let me know if you need any additional information."),
    
    ("Technical Issue", "I'm reporting a technical issue that we've been experiencing with the production system. The problem first appeared yesterday afternoon and has been causing intermittent disruptions to our services. Our technical team has been investigating the root cause, and we've identified several potential factors that might be contributing to the problem. We've implemented temporary workarounds to minimize the impact on our operations, but we need a permanent solution to prevent future occurrences. I've created a detailed incident report that includes error logs, system configurations, and steps we've taken so far. We're coordinating with the vendor's support team to expedite the resolution. I'll keep you updated as we make progress."),
    
    ("Client Feedback", "I wanted to share some valuable feedback we received from our client regarding the recent project delivery. Overall, the feedback was very positive, with particular praise for our attention to detail and timely communication throughout the project lifecycle. However, there were a few areas where the client suggested we could improve, particularly in terms of documentation and post-delivery support. They appreciated our responsiveness and professionalism, which has strengthened our relationship. The client also mentioned they would be interested in discussing future collaboration opportunities. I think this is a great opportunity to refine our processes and potentially expand our business relationship. Let's schedule a debrief meeting to discuss how we can incorporate this feedback into our standard practices."),
    
    ("Team Announcement", "I'm excited to announce some changes to our team structure that will take effect next month. After careful consideration and consultation with leadership, we've decided to reorganize our teams to better align with our strategic objectives. This restructuring will allow us to be more agile and responsive to market demands while maintaining our commitment to quality and innovation. The new structure will create opportunities for professional growth and cross-functional collaboration. We'll be holding a team meeting next week to discuss the details and answer any questions you might have. I'm confident this change will benefit everyone and help us achieve our goals more effectively. Please feel free to reach out if you'd like to discuss this further."),
    
    ("Deadline Extension", "I'm writing to request an extension on the current project deadline. We've encountered some unforeseen challenges that have impacted our timeline, and I believe we need additional time to ensure we deliver a quality product that meets all requirements. The main issues we're facing include resource constraints and some technical complexities that require more thorough investigation. We've already implemented some mitigation strategies, but we need a few more days to complete the work properly. I understand the importance of meeting deadlines, and I want to assure you that we're doing everything possible to minimize the delay. I've prepared a revised timeline that shows how we plan to complete the remaining work. Please let me know if this extension is feasible."),
    
    ("Training Opportunity", "I wanted to bring to your attention an excellent training opportunity that I believe would be beneficial for our team. There's a professional development workshop scheduled for next month that covers advanced techniques in our field. The workshop is being conducted by industry experts and includes hands-on sessions, case studies, and networking opportunities. I think this training would help our team stay current with the latest best practices and enhance our capabilities. The investment in professional development is crucial for maintaining our competitive advantage and ensuring our team members continue to grow in their roles. I've reviewed the curriculum and it aligns well with our current needs and future goals. Please let me know if you'd like me to proceed with registering our team members."),
    
    ("Vendor Proposal", "I've received a proposal from a new vendor that I'd like to discuss with you. They're offering services that could potentially improve our operations and reduce costs. I've done some initial research on their company, and they have a solid track record with similar organizations. Their proposal includes competitive pricing, flexible terms, and additional value-added services that could benefit us. However, I want to make sure we thoroughly evaluate this opportunity before making any commitments. I've prepared a comparison document that outlines the pros and cons of switching vendors versus staying with our current provider. I think it would be valuable to have a discussion about this before we proceed further. Let me know when you're available to review the proposal together."),
    
    ("Performance Review", "I wanted to schedule your annual performance review meeting. This is an important opportunity for us to discuss your achievements over the past year, identify areas for growth, and set goals for the upcoming period. I've been documenting your contributions throughout the year, and I'm impressed with the progress you've made. You've demonstrated strong technical skills and have been a valuable member of the team. During our meeting, we'll discuss your career development plans, potential opportunities for advancement, and how we can support your professional growth. Please come prepared to discuss your accomplishments, challenges you've faced, and your aspirations for the future. I'm looking forward to our conversation.")
]

# Opening salutations
opening_salutations = [
    "Hi {receiver},",
    "Hello {receiver},",
    "Dear {receiver},",
    "Good morning {receiver},",
    "Good afternoon {receiver},",
    "Hi there {receiver},",
    "Hey {receiver},",
    "Greetings {receiver},",
    "Hello {receiver},",
    "Hi {receiver},",
    "Dear {receiver},",
    "Good day {receiver},",
    "Hi {receiver},",
    "Hello {receiver},",
    "Hi {receiver},",
    "Dear {receiver},",
    "Hi {receiver},",
    "Hello {receiver},",
    "Hi {receiver},",
    "Dear {receiver},"
]

# Closing salutations
closing_salutations = [
    ("Best regards,", "{sender}"),
    ("Best,", "{sender}"),
    ("Sincerely,", "{sender}"),
    ("Regards,", "{sender}"),
    ("Thanks,", "{sender}"),
    ("Thank you,", "{sender}"),
    ("Warm regards,", "{sender}"),
    ("Best wishes,", "{sender}"),
    ("Cheers,", "{sender}"),
    ("Take care,", "{sender}"),
    ("Looking forward to your response,", "{sender}"),
    ("Talk soon,", "{sender}"),
    ("All the best,", "{sender}"),
    ("Yours truly,", "{sender}"),
    ("Respectfully,", "{sender}"),
    ("Cordially,", "{sender}"),
    ("With appreciation,", "{sender}"),
    ("Many thanks,", "{sender}"),
    ("Appreciatively,", "{sender}"),
    ("With best regards,", "{sender}")
]

# Generate 100 emails
os.makedirs('emails', exist_ok=True)

for i in range(100):
    sender = random.choice(people)
    receiver = random.choice([p for p in people if p != sender])
    topic, body_template = random.choice(topics)
    
    # Create unique body content (at least 100 words)
    body = body_template
    # Add some variation
    variations = [
        " Additionally, I wanted to emphasize the importance of maintaining open communication channels throughout this process.",
        " It's worth noting that we've seen positive trends in the metrics we've been tracking.",
        " I believe this approach will help us achieve our objectives more efficiently.",
        " We should also consider the long-term implications of these decisions.",
        " I'm confident that with the right resources and support, we can overcome any challenges that arise.",
        " Please don't hesitate to reach out if you need any clarification or have questions.",
        " I look forward to hearing your thoughts and feedback on this matter.",
        " We need to ensure we're all aligned on the priorities and expectations moving forward."
    ]
    body += random.choice(variations)
    
    # Ensure at least 100 words
    while len(body.split()) < 100:
        body += " " + random.choice(variations)
    
    # Format body with line breaks for readability
    # Split by sentences and add line breaks
    import re
    sentences = re.split(r'(?<=[.!?])\s+', body)
    formatted_body = '\n'.join(sentence.strip() for sentence in sentences if sentence.strip())
    
    # Add opening salutation
    opening = random.choice(opening_salutations).format(receiver=receiver['name'].split()[0])
    
    # Add closing salutation
    closing_phrase, closing_name = random.choice(closing_salutations)
    closing = closing_phrase + "\n" + closing_name.format(sender=sender['name'])
    
    email_content = f"""Subject: {topic}

From: {sender['name']} <{sender['email']}>
To: {receiver['name']} <{receiver['email']}>

{opening}

{formatted_body}

{closing}
"""
    
    # Write to file
    filename = f"emails/email_{i+1:03d}.txt"
    with open(filename, 'w') as f:
        f.write(email_content)
    
    print(f"Generated {filename}")

print(f"\nGenerated 100 emails with {len(people)} unique people")
