import numpy as np
import seaborn as sns
import copy


class EntityState(object):
    def __init__(self):
        self.p_pos = None
        self.p_vel = None


class AgentState(EntityState):
    def __init__(self):
        super(AgentState, self).__init__()
        self.c = None


class Action(object):
    def __init__(self):
        self.u = None
        self.c = None


class Wall(object):
    def __init__(self, orient="H", axis_pos=0.0, endpoints=(-1, 1), width=0.1, hard=True):
        self.orient = orient
        self.axis_pos = axis_pos
        self.endpoints = np.array(endpoints)
        self.width = width
        self.hard = hard
        self.color = np.array([0.0, 0.0, 0.0])


class Entity(object):
    def __init__(self):
        self.i = 0
        self.name = ""
        self.size = 0.050
        self.movable = False
        self.collide = True
        self.ghost = False
        self.density = 25.0
        self.color = None
        self.max_speed = None
        self.accel = None
        self.state = EntityState()
        self.initial_mass = 1.0

    @property
    def mass(self):
        return self.initial_mass


class Landmark(Entity):
    def __init__(self):
        super(Landmark, self).__init__()


class Agent(Entity):
    def __init__(self):
        super(Agent, self).__init__()
        self.movable = True
        self.silent = False
        self.blind = False
        self.u_noise = None
        self.c_noise = None
        self.u_range = 1.0
        self.state = AgentState()
        self.action = Action()
        self.action_callback = None


class World(object):
    def __init__(self):
        self.agents = []
        self.landmarks = []
        self.walls = []
        self.dim_c = 0
        self.dim_p = 2
        self.dim_color = 3
        self.dt = 0.1
        self.damping = 0.25
        self.contact_force = 1e2
        self.contact_margin = 1e-3
        self.cache_dists = False
        self.cached_dist_vect = None
        self.cached_dist_mag = None

    @property
    def entities(self):
        return self.agents + self.landmarks

    @property
    def policy_agents(self):
        return [agent for agent in self.agents if agent.action_callback is None]

    @property
    def scripted_agents(self):
        return [agent for agent in self.agents if agent.action_callback is not None]

    def calculate_distances(self):
        if self.cached_dist_vect is None:
            self.cached_dist_vect = np.zeros((len(self.entities), len(self.entities), self.dim_p))
            self.min_dists = np.zeros((len(self.entities), len(self.entities)))
            for ia, entity_a in enumerate(self.entities):
                for ib in range(ia + 1, len(self.entities)):
                    entity_b = self.entities[ib]
                    min_dist = entity_a.size + entity_b.size
                    self.min_dists[ia, ib] = min_dist
                    self.min_dists[ib, ia] = min_dist

        for ia, entity_a in enumerate(self.entities):
            for ib in range(ia + 1, len(self.entities)):
                entity_b = self.entities[ib]
                delta_pos = entity_a.state.p_pos - entity_b.state.p_pos
                self.cached_dist_vect[ia, ib, :] = delta_pos
                self.cached_dist_vect[ib, ia, :] = -delta_pos

        self.cached_dist_mag = np.linalg.norm(self.cached_dist_vect, axis=2)
        self.cached_collisions = self.cached_dist_mag <= self.min_dists

    def assign_agent_colors(self):
        n_dummies = 0
        if hasattr(self.agents[0], "dummy"):
            n_dummies = len([a for a in self.agents if a.dummy])
        n_adversaries = 0
        if hasattr(self.agents[0], "adversary"):
            n_adversaries = len([a for a in self.agents if a.adversary])
        n_good_agents = len(self.agents) - n_adversaries - n_dummies
        dummy_colors = [(0, 0, 0)] * n_dummies
        adv_colors = sns.color_palette("OrRd_d", n_adversaries)
        good_colors = sns.color_palette("GnBu_d", n_good_agents)
        colors = dummy_colors + adv_colors + good_colors
        for color, agent in zip(colors, self.agents):
            agent.color = color

    def step(self):
        for agent in self.scripted_agents:
            agent.action.u = agent.action_callback(
                agent, self
            )
        p_force = [None] * len(self.entities)
        p_force = self.apply_action_force(p_force)
        p_force = self.apply_environment_force(p_force)
        self.integrate_state(p_force)
        for agent in self.agents:
            self.update_agent_state(agent)
        if self.cache_dists:
            self.calculate_distances()

    def apply_action_force(self, p_force):
        for i, agent in enumerate(self.agents):
            if agent.movable:
                noise = (
                    np.random.randn(*agent.action.u.shape) * agent.u_noise if agent.u_noise else 0.0
                )
                p_force[i] = (
                    agent.mass * agent.accel if agent.accel is not None else agent.mass
                ) * agent.action.u + noise
        return p_force

    def apply_environment_force(self, p_force):
        for a, entity_a in enumerate(self.entities):
            for b, entity_b in enumerate(self.entities):
                if b <= a:
                    continue
                [f_a, f_b] = self.get_entity_collision_force(a, b)
                if f_a is not None:
                    if p_force[a] is None:
                        p_force[a] = 0.0
                    p_force[a] = f_a + p_force[a]
                if f_b is not None:
                    if p_force[b] is None:
                        p_force[b] = 0.0
                    p_force[b] = f_b + p_force[b]
            if entity_a.movable:
                for wall in self.walls:
                    wf = self.get_wall_collision_force(entity_a, wall)
                    if wf is not None:
                        if p_force[a] is None:
                            p_force[a] = 0.0
                        p_force[a] = p_force[a] + wf
        return p_force

    def integrate_state(self, p_force):
        for i, entity in enumerate(self.entities):
            if not entity.movable:
                continue
            entity.state.p_vel = entity.state.p_vel * (1 - self.damping)
            if p_force[i] is not None:
                entity.state.p_vel += (p_force[i] / entity.mass) * self.dt
            if entity.max_speed is not None:
                speed = np.sqrt(np.square(entity.state.p_vel[0]) + np.square(entity.state.p_vel[1]))
                if speed > entity.max_speed:
                    entity.state.p_vel = (
                        entity.state.p_vel
                        / np.sqrt(
                            np.square(entity.state.p_vel[0]) + np.square(entity.state.p_vel[1])
                        )
                        * entity.max_speed
                    )

            orig_p_pos = copy.deepcopy(entity.state.p_pos)
            entity.state.p_pos += entity.state.p_vel * self.dt

    def update_agent_state(self, agent):
        if agent.silent:
            agent.state.c = np.zeros(self.dim_c)
        else:
            noise = np.random.randn(*agent.action.c.shape) * agent.c_noise if agent.c_noise else 0.0
            agent.state.c = agent.action.c + noise

    def get_entity_collision_force(self, ia, ib):
        entity_a = self.entities[ia]
        entity_b = self.entities[ib]
        if (not entity_a.collide) or (not entity_b.collide):
            return [None, None]
        if (not entity_a.movable) and (not entity_b.movable):
            return [None, None]
        if entity_a is entity_b:
            return [None, None]
        if self.cache_dists:
            delta_pos = self.cached_dist_vect[ia, ib]
            dist = self.cached_dist_mag[ia, ib]
            dist_min = self.min_dists[ia, ib]
        else:
            delta_pos = entity_a.state.p_pos - entity_b.state.p_pos
            dist = np.sqrt(np.sum(np.square(delta_pos)))
            dist_min = entity_a.size + entity_b.size
        k = self.contact_margin
        penetration = np.logaddexp(0, -(dist - dist_min) / k) * k
        force = self.contact_force * delta_pos / dist * penetration
        if entity_a.movable and entity_b.movable:
            force_ratio = entity_b.mass / entity_a.mass
            force_a = force_ratio * force
            force_b = -(1 / force_ratio) * force
        else:
            force_a = +force if entity_a.movable else None
            force_b = -force if entity_b.movable else None
        return [force_a, force_b]

    def get_wall_collision_force(self, entity, wall):
        if entity.ghost and not wall.hard:
            return None
        if wall.orient == "H":
            prll_dim = 0
            perp_dim = 1
        else:
            prll_dim = 1
            perp_dim = 0
        ent_pos = entity.state.p_pos
        if (
            ent_pos[prll_dim] < wall.endpoints[0] - entity.size
            or ent_pos[prll_dim] > wall.endpoints[1] + entity.size
        ):
            return None
        elif ent_pos[prll_dim] < wall.endpoints[0] or ent_pos[prll_dim] > wall.endpoints[1]:
            if ent_pos[prll_dim] < wall.endpoints[0]:
                dist_past_end = ent_pos[prll_dim] - wall.endpoints[0]
            else:
                dist_past_end = ent_pos[prll_dim] - wall.endpoints[1]
            theta = np.arcsin(dist_past_end / entity.size)
            dist_min = np.cos(theta) * entity.size + 0.5 * wall.width
        else:
            theta = 0
            dist_past_end = 0
            dist_min = entity.size + 0.5 * wall.width

        delta_pos = ent_pos[perp_dim] - wall.axis_pos
        dist = np.abs(delta_pos)
        k = self.contact_margin
        penetration = np.logaddexp(0, -(dist - dist_min) / k) * k
        force_mag = self.contact_force * delta_pos / dist * penetration
        force = np.zeros(2)
        force[perp_dim] = np.cos(theta) * force_mag
        force[prll_dim] = np.sin(theta) * np.abs(force_mag)
        return force
